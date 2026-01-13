import streamlit as st
import datetime
import time
import base64
import pandas as pd
import random
import plotly.express as px
from supabase import create_client
from groq import Groq
from PyPDF2 import PdfReader
import folium
from streamlit_folium import st_folium
import requests
from PIL import Image
import io
import google.generativeai as genai

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def make_pwa_ready():
    """Styles the app to look like a high-quality mobile app"""
    st.markdown("""
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <style>
            footer {visibility: hidden;}
            div.block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
            }
            div.stButton > button {
                width: 100%;
                border-radius: 10px;
                height: 3rem;
                font-weight: 600;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initializes all variables to prevent crashes"""
    defaults = {
        "user": None,
        "user_id": None,
        "feature": "🏠 Home",
        "xp": 0,
        "streak": 0,
        "last_action_date": None,
        "waste_guidelines_text": "",
        "daily_challenges": [],
        "last_challenge_date": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 2. CLIENT CONFIGURATION
# ==========================================
st.set_page_config(page_title="EcoWise AI", page_icon="🌱", layout="wide")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except FileNotFoundError:
    st.error("🚨 Critical: Secrets file not found. Please check .streamlit/secrets.toml")
    st.stop()

@st.cache_resource
def init_clients():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    return supa, groq

supabase, groq_client = init_clients()

# ==========================================
# 3. ROBUST AI LOGIC (Triple Layer)
# ==========================================

def ask_groq(prompt, system_role="You are a helpful Sustainability Expert."):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Logic Error: {str(e)}"

def get_best_gemini_model():
    """Hunts for a working model to fix 404 errors"""
    try:
        # Ask Google what models are actually available
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
                if 'pro' in m.name: return m.name
    except:
        pass
    return "models/gemini-1.5-flash"

def analyze_image_robust(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes))

    # 1. Try Google Gemini (Best)
    if GEMINI_API_KEY:
        try:
            model_name = get_best_gemini_model()
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([
                "Identify this object. Is it recyclable? How to dispose? Be brief.", 
                image_pil
            ])
            return response.text
        except: pass 

    # 2. Try Hugging Face (Backup)
    if HF_TOKEN:
        try:
            API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=5)
            if response.status_code == 200:
                item = response.json()[0]['generated_text']
                advice = ask_groq(f"How to recycle '{item}'?")
                return f"**Detected:** {item}\n\n{advice}"
        except: pass

    return "MANUAL_FALLBACK"

def transcribe_audio(audio_bytes):
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes),
            model="whisper-large-v3",
            response_format="json", language="en", temperature=0.0
        )
        return transcription.text
    except Exception as e: return f"Error: {str(e)}"

def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        return "".join([p.extract_text() for p in pdf.pages])
    except: return None

# ==========================================
# 4. GAMIFICATION & SYNC (FIXED STREAK LOGIC)
# ==========================================
def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    
    # 1. Update XP
    st.session_state.xp += amount
    today = datetime.date.today()
    today_str = str(today)
    
    # 2. Update Streak (Correct Logic)
    last_date_str = st.session_state.last_action_date
    
    if last_date_str != today_str:
        if last_date_str:
            last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
            delta = (today - last_date).days
            
            if delta == 1:
                st.session_state.streak += 1 # Consecutive day
            elif delta > 1:
                st.session_state.streak = 1 # Broken streak
            # If delta == 0 (same day), do nothing to streak
        else:
            st.session_state.streak = 1 # First time ever

        st.session_state.last_action_date = today_str
    
    # 3. Sync to DB
    try:
        supabase.table("user_stats").update({
            "xp": st.session_state.xp, 
            "streak": st.session_state.streak,
            "last_study_date": today_str
        }).eq("user_id", st.session_state.user_id).execute()
        
        supabase.table("study_logs").insert({
            "user_id": st.session_state.user_id, "minutes": amount, "activity_type": activity_name, "date": today_str
        }).execute()
        
        st.toast(f"🌱 +{amount} XP!", icon="🎉")
    except Exception as e:
        print(f"Sync Error: {e}")

def sync_user_stats(uid):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", uid).execute()
        if data.data:
            st.session_state.xp = data.data[0].get('xp', 0)
            st.session_state.streak = data.data[0].get('streak', 0)
            st.session_state.last_action_date = data.data[0].get('last_study_date')
        else:
            supabase.table("user_stats").insert({"user_id": uid, "xp": 0, "streak": 0}).execute()
    except: pass

# ==========================================
# 5. FEATURE RENDERERS (ALL 12 FEATURES)
# ==========================================

def render_home():
    st.title(f"👋 Hi, Eco-Warrior")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", st.session_state.xp)
    c2.metric("Streak", f"{st.session_state.streak}🔥")
    c3.metric("Rank", "Titan" if st.session_state.xp > 1000 else "Rookie")
    
    st.divider()
    st.subheader("🚀 Quick Actions")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Recycling Bot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
        if st.button("🏆 Leaderboard"): navigate_to("🏆 Leaderboard")
    with c2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
        if st.button("👣 Carbon Tracker"): navigate_to("👣 Carbon Tracker")

def render_visual_sorter():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Waste Sorter")
    st.info("Take a photo. AI will tell you how to recycle it.")
    
    t1, t2 = st.tabs(["Camera", "Upload"])
    img_data = None
    with t1:
        cam = st.camera_input("Snap Photo")
        if cam: img_data = cam.getvalue()
    with t2:
        up = st.file_uploader("Or Upload", type=['jpg','png','jpeg'])
        if up: img_data = up.getvalue(); st.image(img_data, width=200)

    if img_data:
        with st.spinner("Analyzing..."):
            res = analyze_image_robust(img_data)
            if res == "MANUAL_FALLBACK":
                st.warning("⚠️ AI busy. Type item name:")
                man = st.text_input("Item Name")
                if man and st.button("Check"):
                    st.markdown(ask_groq(f"Recycle instructions for {man}"))
                    add_xp(15, "Manual Scan")
            else:
                st.success("✅ Identified!")
                st.markdown(res)
                add_xp(15, "Visual Scan")

def render_map():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Interactive Eco-Map")
    st.info("👆 Click map to Pin Location (No typing needed!)")
    
    # Create Map
    m = folium.Map(location=[20.59, 78.96], zoom_start=4)
    pts = supabase.table("map_points").select("*").execute().data
    for p in pts:
        folium.Marker([p['latitude'], p['longitude']], popup=p['name'], icon=folium.Icon(color="green")).add_to(m)
    
    # Render and Capture Click
    map_data = st_folium(m, height=400, width=700)
    
    # Handle Click Logic
    lat, lon = 20.59, 78.96
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"📍 Selected Coordinates: {lat:.4f}, {lon:.4f}")
    
    # Form
    with st.form("pin"):
        st.subheader("Add this Spot")
        n = st.text_input("Location Name")
        t = st.selectbox("Type", ["Recycle Bin", "E-Waste", "Donation Center", "Compost"])
        if st.form_submit_button("📍 Pin Spot"):
            supabase.table("map_points").insert({
                "user_id": st.session_state.user_id, 
                "name": n, "latitude": lat, "longitude": lon, "type": t
            }).execute()
            st.success("Pinned!"); st.rerun()

def render_plastic_calculator():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌊 Plastic Footprint Calculator")
    st.info("Calculate your yearly impact and get a reduction plan.")
    
    c1, c2 = st.columns(2)
    with c1:
        b = st.slider("Bottles (per week)", 0, 50, 5)
        bg = st.slider("Plastic Bags (per week)", 0, 50, 5)
    with c2:
        w = st.slider("Wrappers/Packets (per week)", 0, 50, 10)
        c = st.slider("Disposable Cups (per week)", 0, 20, 2)
    
    # Calculation
    kg = ((b*12 + bg*5 + w*2 + c*10) * 52) / 1000
    st.metric("Your Annual Plastic Waste", f"{kg:.2f} kg")
    
    if kg < 5: st.success("🌟 Low Impact!")
    elif kg < 15: st.warning("⚠️ Moderate Impact")
    else: st.error("🚨 High Impact - Needs Action!")
    
    if st.button("📉 Get Reduction Strategy"):
        st.markdown(ask_groq(f"I generate {kg}kg plastic/year. Give me 3 strict tips to reduce this."))
        add_xp(20, "Plastic Audit")

def render_forest():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Virtual Forest")
    
    trees = st.session_state.xp // 100
    remainder = 100 - (st.session_state.xp % 100)
    
    st.metric("Trees Planted", trees, delta=f"Next tree in {remainder} pts")
    
    if trees == 0: 
        st.markdown("# 🌱")
        st.caption("A seedling! Keep recycling to grow it.")
    elif trees < 5: 
        st.markdown(f"# {'🌲 ' * trees}")
        st.caption("A small grove is forming.")
    else: 
        st.markdown(f"# {'🌳 ' * trees}")
        st.success("You have a lush forest!")
        
    st.write("### 📜 Impact Log")
    logs = supabase.table("study_logs").select("*").eq("user_id", st.session_state.user_id).order("date", desc=True).limit(5).execute().data
    for l in logs:
        st.text(f"{l['date']} - {l['activity_type']} (+{l['minutes']} pts)")

def render_upcycling():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎨 Trash-to-Treasure")
    item = st.text_input("I have an old...")
    if item and st.button("Get Ideas"):
        st.markdown(ask_groq(f"3 creative DIY upcycling ideas for {item}. Be brief."))
        add_xp(25, "Upcycling")

def render_menu():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🥗 Low-Carbon Menu")
    c = st.selectbox("Cuisine", ["Indian", "Italian", "Mexican", "Asian"])
    if st.button("Plan Meal"):
        st.markdown(ask_groq(f"Suggest a low-carbon {c} meal plan. Explain why it's eco-friendly."))
        add_xp(20, "Menu Plan")

def render_voice():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Mode")
    aud = st.audio_input("Speak Question")
    if aud:
        txt = transcribe_audio(aud)
        st.write(f"You: {txt}")
        st.markdown(ask_groq(txt))
        add_xp(10, "Voice Query")

def render_chat():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Recycle Assistant")
    up = st.file_uploader("Upload Rules PDF", type=['pdf'])
    if up: st.session_state.waste_guidelines_text = extract_text_from_pdf(up)
    q = st.chat_input("Ask about recycling...")
    if q: st.markdown(ask_groq(q + (st.session_state.waste_guidelines_text or "")))
    add_xp(5, "Chat")

def render_mistake():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    m = st.text_input("I threw...")
    b = st.selectbox("Into...", ["Recycle Bin", "Compost", "Trash"])
    if st.button("Explain Impact"): 
        st.markdown(ask_groq(f"I put {m} into {b}. Explain environmental consequence."))
        add_xp(10, "Mistake Check")

def render_leaderboard():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🏆 Global Leaderboard")
    try:
        data = supabase.table("user_stats").select("*").order("xp", desc=True).limit(10).execute().data
        df = pd.DataFrame(data)
        st.dataframe(df[['xp', 'streak']], use_container_width=True)
    except: st.error("Unavailable")

def render_carbon_tracker():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    
    st.header("👣 Carbon Footprint Saver")
    st.info("Calculate how much CO₂ you saved by choosing eco-friendly transport today.")
    
    # 1. Inputs
    with st.form("carbon_calc"):
        mode = st.selectbox("How did you travel today?", ["Walk / Bicycle", "Bus / Metro / Train", "Carpool", "Electric Vehicle"])
        hours = st.number_input("Travel Duration (Hours)", min_value=0.1, max_value=24.0, value=1.0, step=0.5)
        
        submitted = st.form_submit_button("🌱 Calculate Savings")
        
    if submitted:
        # Savings Logic (Compared to a standard petrol car emitting ~200g CO2/km at 40km/h)
        # Standard Car = ~8 kg CO2 per hour of driving
        
        savings_per_hour = 0
        if mode == "Walk / Bicycle":
            savings_per_hour = 8.0  # You produced 0, so you saved all 8kg
        elif mode == "Bus / Metro / Train":
            savings_per_hour = 5.5  # Public transport is much cleaner per person
        elif mode == "Electric Vehicle":
            savings_per_hour = 4.0  # cleaner, but still uses energy
        elif mode == "Carpool":
            savings_per_hour = 3.0  # Sharing the ride saves a portion
            
        total_saved = savings_per_hour * hours
        
        # 2. Display Results
        st.divider()
        c_res, c_msg = st.columns(2)
        with c_res:
            st.metric("CO₂ Emissions Prevented", f"{total_saved:.2f} kg", delta="Eco-Impact")
        with c_msg:
            st.success(f"Great job! By choosing to {mode} for {hours} hours, you prevented {total_saved:.1f} kg of carbon from entering the atmosphere.")
            
        # 3. Gamification
        xp_earned = int(total_saved * 10) # 10 points per kg saved
        add_xp(xp_earned, f"Transport: {mode}")

# ==========================================
# 6. MAIN APP LOOP
# ==========================================
def main():
    init_session_state()
    make_pwa_ready()
    
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        t1, t2 = st.tabs(["Login", "Sign Up"])
        with t1:
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Login"): 
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.session_state.user = res.user; st.session_state.user_id = res.user.id
                    sync_user_stats(res.user.id); st.rerun()
                except Exception as err: st.error(str(err))
        with t2:
            e2 = st.text_input("Email (New)")
            p2 = st.text_input("Password (New)", type="password")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: 
                        supabase.table("user_stats").insert({"user_id": res.user.id}).execute()
                        st.success("Created! Please Login.")
                except Exception as err: st.error(str(err))
        return

    with st.sidebar:
        st.title("EcoWise")
        st.caption(f"User: {st.session_state.user.email}")
        st.divider()
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycling Bot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
        if st.button("🏆 Leaderboard"): navigate_to("🏆 Leaderboard")
        if st.button("👣 Carbon Tracker"): navigate_to("👣 Carbon Tracker")
        if st.button("❌ Mistake Fixer"): navigate_to("❌ Mistake Explainer")
        st.divider()
        if st.button("🚪 Logout"): 
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "🎙️ Voice Mode": render_voice()
    elif f == "♻️ Recycle Assistant": render_chat()
    elif f == "❌ Mistake Explainer": render_mistake()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🌊 Plastic Calculator": render_plastic_calculator()
    elif f == "🎨 Upcycling Station": render_upcycling()
    elif f == "🥗 Eco-Menu Planner": render_menu()
    elif f == "🌳 My Forest": render_forest()
    elif f == "🏆 Leaderboard": render_leaderboard()
    elif f == "👣 Carbon Tracker": render_carbon_tracker()

if __name__ == "__main__":
    main()

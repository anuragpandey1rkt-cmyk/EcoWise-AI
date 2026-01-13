import streamlit as st
import datetime
from datetime import date, timedelta
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
# 1. CONFIGURATION & MOBILE STYLES
# ==========================================
st.set_page_config(
    page_title="EcoWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed by default for mobile feel
)

def apply_mobile_styles():
    st.markdown("""
        <style>
            /* Hide Streamlit Header/Footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Mobile Container Optimization */
            div.block-container {
                padding-top: 1rem;
                padding-bottom: 5rem;
                max-width: 100%;
            }
            
            /* Big Touchable Buttons */
            div.stButton > button {
                width: 100%;
                border-radius: 15px;
                height: 3.5rem;
                font-weight: 600;
                font-size: 16px;
                border: none;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.1s;
                background-color: #ffffff;
                color: #2E7D32; /* Eco Green Text */
                border: 1px solid #2E7D32;
            }
            div.stButton > button:active {
                transform: scale(0.98);
                background-color: #2E7D32;
                color: white;
            }

            /* Metric Cards */
            div[data-testid="stMetric"] {
                background-color: #f1f8e9; /* Light Green Bg */
                border-radius: 12px;
                padding: 10px;
                border: 1px solid #c5e1a5;
                text-align: center;
            }

            /* Input Fields styling */
            div[data-testid="stTextInput"] > div > div > input {
                border-radius: 10px;
            }
        </style>
        
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="theme-color" content="#2E7D32">
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    """, unsafe_allow_html=True)

# ==========================================
# 2. CLIENTS & SECRETS
# ==========================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except FileNotFoundError:
    st.error("Secrets not found. Please set up .streamlit/secrets.toml")
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
# 3. SESSION STATE
# ==========================================
def init_session_state():
    defaults = {
        "user": None,
        "user_id": None,
        "feature": "🏠 Home",
        "xp": 0,
        "streak": 0,
        "last_action_date": None,
        "waste_guidelines_text": "",
        "daily_challenges": [],
        "last_challenge_date": None,
        "chat_history": [] # For Chatbot
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 4. ROBUST AI ENGINES
# ==========================================
def ask_groq(prompt, system_role="You are a helpful Sustainability Expert."):
    """Uses Groq (Llama 3) for text logic"""
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
    """Hunts for a working Vision model available to your key"""
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
                if 'pro' in m.name: return m.name
    except: pass
    return "gemini-1.5-flash"

def analyze_image_robust(image_bytes):
    """TRIPLE LAYER SAFETY SYSTEM"""
    image_pil = Image.open(io.BytesIO(image_bytes))

    # --- LAYER 1: GOOGLE GEMINI ---
    if GEMINI_API_KEY:
        try:
            model_name = get_best_gemini_model()
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([
                "Identify this object exactly. Is it recyclable, compostable, or trash? Be brief and give strict disposal instructions.", 
                image_pil
            ])
            return response.text
        except: pass

    # --- LAYER 2: HUGGING FACE ---
    if HF_TOKEN:
        try:
            API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=5)
            
            if response.status_code == 200:
                prediction = response.json()
                if isinstance(prediction, list) and 'generated_text' in prediction[0]:
                    item_name = prediction[0]['generated_text']
                    advice = ask_groq(f"How do I recycle '{item_name}'? Be strict.")
                    return f"**Detected:** {item_name.title()}\n\n{advice}"
        except: pass

    # --- LAYER 3: FAIL SIGNAL ---
    return "MANUAL_FALLBACK"

def transcribe_audio(audio_bytes):
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        return transcription.text
    except Exception as e:
        return f"Audio Error: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except: return None

# ==========================================
# 5. BACKEND LOGIC (STREAK FIX)
# ==========================================

def sync_user_stats(user_id):
    """Syncs data from DB when user logs in."""
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", user_id).execute()
        if data.data:
            stats = data.data[0]
            st.session_state.xp = stats.get('xp', 0)
            st.session_state.streak = stats.get('streak', 0)
            st.session_state.last_action_date = stats.get('last_study_date')
        else:
            # First time user
            supabase.table("user_stats").insert({"user_id": user_id, "xp": 0, "streak": 1, "last_study_date": str(date.today())}).execute()
            st.session_state.xp = 0
            st.session_state.streak = 1
            st.session_state.last_action_date = str(date.today())
    except Exception as e:
        print(f"Sync Error: {e}")

def add_xp(amount, activity_name):
    """
    Adds XP and Updates Streak intelligently.
    FIX: Streak only increases if last action was Yesterday.
    """
    if not st.session_state.user_id: return
    
    # 1. Update XP in Session
    st.session_state.xp += amount
    
    # 2. Date Logic for Streak
    today = date.today()
    last_date_str = st.session_state.last_action_date
    
    new_streak = st.session_state.streak
    
    if last_date_str:
        last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
        diff = (today - last_date).days
        
        if diff == 0:
            pass # Same day, keep streak same
        elif diff == 1:
            new_streak += 1 # Consecutive day, increase streak
        else:
            new_streak = 1 # Missed a day, reset streak
    else:
        new_streak = 1 # Should not happen if sync works, but safety net

    # 3. Update Session
    st.session_state.streak = new_streak
    st.session_state.last_action_date = str(today)

    # 4. Push to DB
    try:
        supabase.table("user_stats").update({
            "xp": st.session_state.xp,
            "streak": new_streak,
            "last_study_date": str(today)
        }).eq("user_id", st.session_state.user_id).execute()
        
        supabase.table("study_logs").insert({
            "user_id": st.session_state.user_id, 
            "minutes": amount, 
            "activity_type": activity_name, 
            "date": str(today)
        }).execute()
        
        st.toast(f"🌱 +{amount} XP! Streak: {new_streak} 🔥", icon="🎉")
    except Exception as e:
        st.error(f"Sync Error: {e}")

# ==========================================
# 6. FEATURE RENDERERS
# ==========================================

def render_home():
    st.write("") 
    st.markdown(f"### 👋 Hi, Eco-Warrior")
    
    # Dashboard Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Points", st.session_state.xp)
    c2.metric("🔥 Streak", f"{st.session_state.streak} Days")
    
    # Determine Rank
    xp = st.session_state.xp
    rank = "Seedling"
    if xp > 100: rank = "Sapling"
    if xp > 500: rank = "Guardian"
    if xp > 1000: rank = "Titan"
    c3.metric("🏆 Rank", rank)
    
    st.divider()
    
    # Action Grid
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("👣 Carbon Calc"): navigate_to("👣 Carbon Tracker")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
    with col2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
    
    if st.button("🥗 Eco-Menu Planner"): navigate_to("🥗 Eco-Menu Planner")
    if st.button("📊 Leaderboard"): navigate_to("📊 Leaderboard")
    if st.button("❌ Mistake Explainer"): navigate_to("❌ Mistake Explainer")

def render_visual_sorter():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Waste Sorter")
    st.info("Identify trash instantly using Smart AI.")
    
    tab1, tab2 = st.tabs(["📸 Camera", "📂 Upload"])
    img_data = None

    with tab1:
        cam_img = st.camera_input("Snap Photo")
        if cam_img: img_data = cam_img.getvalue()
    with tab2:
        up_img = st.file_uploader("Select Image", type=['jpg','jpeg','png'])
        if up_img: img_data = up_img.getvalue(); st.image(img_data, width=300)

    if img_data:
        with st.spinner("Analyzing..."):
            res = analyze_image_robust(img_data)
            
            if res == "MANUAL_FALLBACK":
                st.warning("⚠️ Vision AI busy. Describe item:")
                man = st.text_input("Item Name", key="manual_fix")
                if man and st.button("Check"):
                    st.markdown(ask_groq(f"Recycle instructions for {man}"))
                    add_xp(15, "Manual Scan")
            else:
                st.success("✅ Analysis Complete!")
                st.markdown(res)
                add_xp(15, "Visual Scan")

def render_recycle_assistant():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Eco-Chatbot")
    st.caption("Ask anything about recycling or sustainability!")

    # Context Loader
    with st.expander("📄 Upload Local Guidelines (PDF)"):
        f = st.file_uploader("PDF", type=['pdf'])
        if f: 
            st.session_state.waste_guidelines_text = extract_text_from_pdf(f)
            st.success("Guidelines Loaded! I'll use them to answer.")

    # Chat Interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("How do I recycle batteries?"):
        # User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ctx = st.session_state.waste_guidelines_text or ""
                full_prompt = f"Context: {ctx[:4000]}\n\nUser Question: {prompt}"
                response = ask_groq(full_prompt)
                st.write(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        add_xp(5, "Chat Query")

def render_carbon_tracker():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("👣 Advanced Carbon Tracker")
    st.info("Calculate how much CO2 you save by choosing green transport.")
    
    # Improved Inputs
    mode = st.selectbox("How did you travel?", 
                        ["Walk / Bicycle (0g CO2)", "Bus / Train (Low CO2)", "Electric Vehicle (Low CO2)", "Car Share"])
    dist = st.number_input("Distance traveled (km)", min_value=1.0, value=5.0)
    
    # Calc Logic (Comparison vs Single Gas Car ~ 150g/km)
    savings = 0.0
    if "Walk" in mode: savings = (0.150 * dist)
    elif "Bus" in mode: savings = (0.100 * dist) # Bus is efficient per person
    elif "Electric" in mode: savings = (0.100 * dist)
    elif "Share" in mode: savings = (0.075 * dist)
    
    if st.button("🌱 Calculate Impact"):
        st.metric("CO2 Saved vs Driving Alone", f"{savings:.3f} kg")
        st.success(f"Great job! You prevented {savings:.3f} kg of CO2 from entering the atmosphere.")
        add_xp(int(savings * 100) + 10, "Carbon Saved")

def render_virtual_forest():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Virtual Forest")
    st.info("Your actions grow digital trees!")
    
    trees = st.session_state.xp // 100
    st.metric("Trees Planted", trees)
    
    # Forest Visualization
    if trees == 0: 
        st.markdown("<h1 style='text-align: center;'>🌱</h1>", unsafe_allow_html=True)
        st.caption("Just a seedling. Earn 100 XP to grow a tree!")
    elif trees < 10:
        forest = "🌲 " * trees
        st.markdown(f"<h1 style='text-align: center;'>{forest}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center;'>🌳🌳🌳 FOREST UNLOCKED 🌳🌳🌳</h1>", unsafe_allow_html=True)
        
    st.write("### Recent Activity")
    logs = supabase.table("study_logs").select("*").eq("user_id", st.session_state.user_id).order("date", desc=True).limit(3).execute().data
    if logs:
        for l in logs:
            st.text(f"📅 {l['date']} | {l['activity_type']}")

def render_map():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    
    m = folium.Map(location=[20.59, 78.96], zoom_start=4)
    pts = supabase.table("map_points").select("*").execute().data
    for p in pts:
        folium.Marker([p['latitude'], p['longitude']], popup=p['name'], icon=folium.Icon(color="green")).add_to(m)

    map_data = st_folium(m, height=350, width="100%")
    
    lat, lon = 20.59, 78.96
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.caption(f"📍 Selected: {lat:.4f}, {lon:.4f}")

    with st.form("pin"):
        st.write("Add this spot?")
        n = st.text_input("Location Name")
        t = st.selectbox("Type", ["Recycle Bin", "E-Waste", "Donation"])
        if st.form_submit_button("📍 Pin Spot"):
            supabase.table("map_points").insert({"user_id": st.session_state.user_id, "name": n, "latitude": lat, "longitude": lon, "type": t}).execute()
            st.success("Pinned!")
            time.sleep(1); st.rerun()

def render_voice_mode():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Ask me anything")
    if audio:
        with st.spinner("Listening..."):
            txt = transcribe_audio(audio)
            st.write(f"You: {txt}")
            res = ask_groq(txt)
            st.markdown(f"**AI:** {res}")
            add_xp(10, "Voice Query")

def render_plastic_calculator():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌊 Plastic Footprint")
    
    b = st.slider("Bottles/Week", 0, 50, 5)
    bg = st.slider("Bags/Week", 0, 50, 5)
    w = st.slider("Wrappers/Week", 0, 50, 10)
    
    kg = ((b*12 + bg*5 + w*2) * 52) / 1000
    st.metric("Yearly Plastic Waste", f"{kg:.2f} kg")
    
    if st.button("📉 Get Reduction Plan"):
        st.markdown(ask_groq(f"I use {kg}kg plastic/year. Give 3 tips to reduce."))
        add_xp(20, "Audit")

def render_upcycling_station():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎨 Trash-to-Treasure")
    
    item = st.text_input("I have an old...")
    if item and st.button("Get DIY Ideas"):
        st.markdown(ask_groq(f"3 creative DIY upcycling ideas for {item}. Be brief."))
        add_xp(25, "Upcycling")

def render_sustainable_menu():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🥗 Eco-Menu Planner")
    
    c = st.selectbox("Cuisine", ["Indian", "Italian", "Mexican", "Asian"])
    if st.button("Generate Menu"):
        st.markdown(ask_groq(f"Suggest a low-carbon {c} meal plan. Explain why it's eco-friendly."))
        add_xp(20, "Menu Plan")

def render_leaderboard():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🏆 Global Leaderboard")
    try:
        data = supabase.table("user_stats").select("*").order("xp", desc=True).limit(10).execute().data
        if data:
            df = pd.DataFrame(data)
            df['User'] = df['user_id'].apply(lambda x: "You" if x == st.session_state.user_id else f"User {x[:4]}..")
            st.dataframe(df[['User', 'xp', 'streak']], use_container_width=True)
        else: st.info("Loading...")
    except: st.error("Leaderboard unavailable.")

def render_mistake_explainer():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    m = st.text_input("I threw...")
    b = st.selectbox("Into...", ["Recycle", "Compost", "Trash"])
    if st.button("Explain Impact"):
        st.markdown(ask_groq(f"I put {m} into {b}. Is that bad?"))
        add_xp(10, "Learning")

# ==========================================
# 7. MAIN APP LOOP
# ==========================================
def main():
    apply_mobile_styles()
    
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
            e2 = st.text_input("Email (Sign Up)")
            p2 = st.text_input("Password (Sign Up)", type="password")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: 
                        supabase.table("user_stats").insert({"user_id": res.user.id, "streak": 1, "last_study_date": str(date.today())}).execute()
                        st.success("Account created! Please Login.")
                except Exception as err: st.error(str(err))
        return

    # Sidebar Navigation
    with st.sidebar:
        st.title("EcoWise")
        st.write(f"👤 {st.session_state.user.email}")
        st.divider()
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Eco-Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("👣 Carbon Calc"): navigate_to("👣 Carbon Tracker")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
        st.divider()
        if st.button("🚪 Logout"): 
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    # Routing
    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "🎙️ Voice Mode": render_voice_mode()
    elif f == "♻️ Recycle Assistant": render_recycle_assistant()
    elif f == "❌ Mistake Explainer": render_mistake_explainer()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🌊 Plastic Calculator": render_plastic_calculator()
    elif f == "🎨 Upcycling Station": render_upcycling_station()
    elif f == "🥗 Eco-Menu Planner": render_sustainable_menu()
    elif f == "🌳 My Forest": render_virtual_forest()
    elif f == "👣 Carbon Tracker": render_carbon_tracker()
    elif f == "📊 Leaderboard": render_leaderboard()

if __name__ == "__main__":
    main()

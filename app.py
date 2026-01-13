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
    initial_sidebar_state="collapsed"
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
                border-radius: 12px;
                height: 3.2rem;
                font-weight: 600;
                border: 1px solid #e0e0e0;
                box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.1s;
                background-color: #ffffff;
                color: #2E7D32;
            }
            div.stButton > button:active {
                transform: scale(0.98);
                background-color: #f1f8e9;
            }

            /* Profile Sidebar Styles */
            [data-testid="stSidebarUserContent"] {
                padding-top: 2rem;
            }
            .profile-box {
                background-color: #e8f5e9;
                padding: 15px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
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
        "chat_history": []
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
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
                if 'pro' in m.name: return m.name
    except: pass
    return "gemini-1.5-flash"

def analyze_image_robust(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Layer 1: Gemini
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

    # Layer 2: Hugging Face
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
# 5. BACKEND LOGIC
# ==========================================
def sync_user_stats(user_id):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", user_id).execute()
        if data.data:
            stats = data.data[0]
            st.session_state.xp = stats.get('xp', 0)
            st.session_state.streak = stats.get('streak', 0)
            st.session_state.last_action_date = stats.get('last_study_date')
        else:
            supabase.table("user_stats").insert({"user_id": user_id, "xp": 0, "streak": 1, "last_study_date": str(date.today())}).execute()
            st.session_state.xp = 0; st.session_state.streak = 1
    except: pass

def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    st.session_state.xp += amount
    today = str(date.today())
    
    # Streak Logic
    new_streak = st.session_state.streak
    if st.session_state.last_action_date:
        last = datetime.datetime.strptime(st.session_state.last_action_date, "%Y-%m-%d").date()
        diff = (date.today() - last).days
        if diff == 1: new_streak += 1
        elif diff > 1: new_streak = 1
    
    st.session_state.streak = new_streak
    st.session_state.last_action_date = today

    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp, "streak": new_streak, "last_study_date": today}).eq("user_id", st.session_state.user_id).execute()
        supabase.table("study_logs").insert({"user_id": st.session_state.user_id, "minutes": amount, "activity_type": activity_name, "date": today}).execute()
        st.toast(f"🌱 +{amount} XP!", icon="🎉")
    except: pass

# ==========================================
# 6. FEATURE RENDERERS
# ==========================================

def render_home():
    st.write(""); st.markdown(f"### 👋 Welcome back")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Points", st.session_state.xp)
    c2.metric("🔥 Streak", f"{st.session_state.streak} Days")
    rank = "Seedling" if st.session_state.xp < 100 else ("Guardian" if st.session_state.xp < 500 else "Titan")
    c3.metric("🏆 Rank", rank)
    
    st.divider()
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Eco-Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("👣 Carbon Calc"): navigate_to("👣 Carbon Tracker")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
    with col2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")

    if st.button("🥗 Eco-Menu Planner"): navigate_to("🥗 Eco-Menu Planner")

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
    with st.expander("📄 Upload Local Guidelines (PDF)"):
        f = st.file_uploader("PDF", type=['pdf'])
        if f: st.session_state.waste_guidelines_text = extract_text_from_pdf(f); st.success("Guidelines Loaded!")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("How do I recycle batteries?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ctx = st.session_state.waste_guidelines_text or ""
                response = ask_groq(f"Context: {ctx[:4000]}\n\nUser Question: {prompt}")
                st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        add_xp(5, "Chat Query")

def render_carbon_tracker():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("👣 Carbon Tracker")
    mode = st.selectbox("Transport Mode", ["Walk/Bike (0g)", "Bus/Train (Low)", "Electric Car (Low)", "Carpool"])
    dist = st.number_input("Distance (km)", min_value=1.0, value=5.0)
    savings = 0.150 * dist if "Walk" in mode else (0.100 * dist)
    if st.button("🌱 Log Trip"):
        st.success(f"You saved {savings:.3f} kg CO2 vs driving alone!")
        add_xp(int(savings * 100) + 10, "Carbon Saved")

def render_virtual_forest():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Virtual Forest")
    trees = st.session_state.xp // 100
    st.metric("Trees Planted", trees)
    if trees < 1: st.markdown("### 🌱")
    elif trees < 10: st.markdown(f"### {'🌲 ' * trees}")
    else: st.markdown(f"### 🌳🌳 FOREST UNLOCKED 🌳🌳")

def render_map():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    m = folium.Map(location=[20.59, 78.96], zoom_start=4)
    pts = supabase.table("map_points").select("*").execute().data
    for p in pts: folium.Marker([p['latitude'], p['longitude']], popup=p['name']).add_to(m)
    map_data = st_folium(m, height=350, width="100%")
    
    lat, lon = 20.59, 78.96
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.caption(f"📍 Selected: {lat:.4f}, {lon:.4f}")
    
    with st.form("pin"):
        n = st.text_input("Name"); t = st.selectbox("Type", ["Bin", "E-Waste", "Donation"])
        if st.form_submit_button("📍 Pin"):
            supabase.table("map_points").insert({"user_id": st.session_state.user_id, "name": n, "latitude": lat, "longitude": lon, "type": t}).execute()
            st.success("Pinned!"); time.sleep(1); st.rerun()

def render_voice_mode():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Ask me anything")
    if audio:
        txt = transcribe_audio(audio)
        st.write(f"You: {txt}")
        st.markdown(ask_groq(txt))
        add_xp(10, "Voice Query")

def render_plastic_calculator():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌊 Plastic Footprint")
    b = st.slider("Bottles/Week", 0, 50, 5); bg = st.slider("Bags/Week", 0, 50, 5)
    kg = ((b*12 + bg*5) * 52) / 1000
    st.metric("Yearly Plastic", f"{kg:.2f} kg")
    if st.button("📉 Reduction Plan"):
        st.markdown(ask_groq(f"I use {kg}kg plastic/year. Tips to reduce?"))
        add_xp(20, "Audit")

def render_upcycling_station():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎨 Trash-to-Treasure")
    item = st.text_input("I have an old...")
    if item and st.button("Get Ideas"):
        st.markdown(ask_groq(f"DIY upcycling ideas for {item}."))
        add_xp(25, "Upcycling")

def render_sustainable_menu():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🥗 Eco-Menu Planner")
    c = st.selectbox("Cuisine", ["Indian", "Italian", "Mexican", "Asian"])
    if st.button("Generate Menu"):
        st.markdown(ask_groq(f"Low-carbon {c} meal plan."))
        add_xp(20, "Menu Plan")

def render_leaderboard():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🏆 Leaderboard")
    try:
        data = supabase.table("user_stats").select("*").order("xp", desc=True).limit(10).execute().data
        if data:
            df = pd.DataFrame(data)
            df['User'] = df['user_id'].apply(lambda x: "You" if x == st.session_state.user_id else f"User {x[:4]}..")
            st.dataframe(df[['User', 'xp', 'streak']], use_container_width=True)
    except: st.error("Unavailable.")

def render_mistake_explainer():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    m = st.text_input("I threw...")
    b = st.selectbox("Into...", ["Recycle", "Trash"])
    if st.button("Check"):
        st.markdown(ask_groq(f"I put {m} into {b}. Bad?"))
        add_xp(10, "Learning")

def render_settings():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("⚙️ Settings")
    st.toggle("🔔 Enable Notifications", value=True)
    st.toggle("🌙 Dark Mode", value=False)
    st.toggle("📍 Share Location", value=True)
    st.caption("App Version 2.0 (Mobile Optimized)")

# ==========================================
# 7. MAIN APP LOOP & SIDEBAR
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
                        st.success("Account created! Login.")
                except Exception as err: st.error(str(err))
        return

    # --- SIDEBAR (PROFILE & MENU) ---
    with st.sidebar:
        # Profile Section
        user_name = st.session_state.user.email.split("@")[0].capitalize()
        xp = st.session_state.xp
        rank = "Seedling" if xp < 100 else ("Guardian" if xp < 500 else "Titan")
        
        st.markdown(f"""
        <div class="profile-box">
            <h3>👤 {user_name}</h3>
            <p>🏆 Rank: <b>{rank}</b></p>
            <p>⭐ XP: <b>{xp}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 🧭 Menu")
        if st.button("🏠 Dashboard"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("👣 Carbon Tracker"): navigate_to("👣 Carbon Tracker")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
        
        st.divider()
        st.markdown("### ⚙️ Account")
        if st.button("⚙️ Settings"): navigate_to("⚙️ Settings")
        if st.button("🚪 Logout"): 
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    # Routing
    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "♻️ Recycle Assistant": render_recycle_assistant()
    elif f == "👣 Carbon Tracker": render_carbon_tracker()
    elif f == "🌳 My Forest": render_virtual_forest()
    elif f == "⚙️ Settings": render_settings()
    elif f == "🎙️ Voice Mode": render_voice_mode()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🌊 Plastic Calculator": render_plastic_calculator()
    elif f == "🎨 Upcycling Station": render_upcycling_station()
    elif f == "🥗 Eco-Menu Planner": render_sustainable_menu()
    elif f == "📊 Leaderboard": render_leaderboard()
    elif f == "❌ Mistake Explainer": render_mistake_explainer()

if __name__ == "__main__":
    main()

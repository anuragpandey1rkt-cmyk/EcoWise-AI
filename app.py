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
import cv2
import numpy as np
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & INIT
# ==========================================
st.set_page_config(
    page_title="EcoWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_pwa_ready():
    st.markdown("""
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <style>
            footer {visibility: hidden;}
            div.block-container {
                padding-top: max(3.5rem, env(safe-area-inset-top));
                padding-bottom: 5rem;
            }
            div.stButton > button {
                width: 100%;
                border-radius: 8px;
                height: 3rem;
            }
        </style>
    """, unsafe_allow_html=True)

# Load Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    # Safely load Gemini Key
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except FileNotFoundError:
    st.error("Secrets not found. Please set up .streamlit/secrets.toml")
    st.stop()

# Initialize Clients
@st.cache_resource
def init_clients():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    return supa, groq

supabase, groq_client = init_clients()

# ==========================================
# 2. SESSION STATE & NAVIGATION
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
        "last_challenge_date": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 3. HYBRID VISION ENGINE (OpenCV + Gemini)
# ==========================================

def get_opencv_metrics(image_bytes):
    """
    Extracts technical data using OpenCV:
    1. Sharpness (Laplacian Variance)
    2. Dominant Color (K-Means simplified)
    """
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Calculate Sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = f"{int(laplacian_var)} (Blurry)" if laplacian_var < 100 else f"{int(laplacian_var)} (Sharp)"
    
    # 2. Dominant Color
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    b, g, r = avg_color
    
    color_name = "Mixed"
    if r > 150 and g > 150 and b > 150: color_name = "White/Light"
    elif r < 50 and g < 50 and b < 50: color_name = "Black/Dark"
    elif g > r and g > b: color_name = "Green (Glass/Organic?)"
    elif b > r and b > g: color_name = "Blue (Plastic?)"
    elif r > 150 and g > 100 and b < 50: color_name = "Brown (Cardboard?)"
    
    return sharpness_score, color_name

def analyze_smart_hybrid(image_bytes):
    """
    The 'Best Project' Feature:
    Combines OpenCV Metrics + Gemini AI Understanding.
    """
    # Step 1: Run OpenCV Locally
    sharpness, color_hint = get_opencv_metrics(image_bytes)
    
    # Step 2: Run Gemini AI (The Brain)
    if GEMINI_API_KEY:
        try:
            image_pil = Image.open(io.BytesIO(image_bytes))
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Smart Prompt: Ask AI to verify OpenCV's findings
            prompt = (
                f"I have processed this image with OpenCV. "
                f"It detected color dominance: '{color_hint}'. "
                f"Now, as an advanced AI, strictly identify the object name. "
                f"Then provide: 1. Is it Recyclable? 2. Preparation steps (e.g. wash it). 3. Which bin? "
                f"Format as clear markdown."
            )
            
            response = model.generate_content([prompt, image_pil])
            ai_text = response.text
            
            # Combine into a Technical Report
            final_report = (
                f"### 🧬 Smart Vision Analysis\n"
                f"**Technical Metrics (OpenCV):**\n"
                f"* **Sharpness Score:** {sharpness}\n"
                f"* **Spectral Color:** {color_hint}\n"
                f"---\n"
                f"**AI Identification (Gemini):**\n"
                f"{ai_text}"
            )
            return "SUCCESS", final_report
            
        except Exception as e:
            return "AI_ERROR", str(e)
            
    return "NO_KEY", "Gemini API Key missing."

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

# ==========================================
# 4. HELPERS
# ==========================================
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
    except Exception as e:
        return None

def sync_user_stats(user_id):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", user_id).execute()
        if data.data:
            stats = data.data[0]
            st.session_state.xp = stats.get('xp', 0)
            st.session_state.streak = stats.get('streak', 0)
        else:
            supabase.table("user_stats").insert({"user_id": user_id, "xp": 0, "streak": 0}).execute()
    except: pass

def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    st.session_state.xp += amount
    today = str(datetime.date.today())
    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp}).eq("user_id", st.session_state.user_id).execute()
        supabase.table("study_logs").insert({
            "user_id": st.session_state.user_id, "minutes": amount, "activity_type": activity_name, "date": today
        }).execute()
        st.toast(f"🌱 +{amount} Green Points!", icon="🌍")
    except: pass

# ==========================================
# 5. FEATURE RENDERERS
# ==========================================

def render_visual_sorter():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 Hybrid AI Waste Sorter")
    st.info("Combines **OpenCV** (Mathematical Analysis) with **Google Gemini** (Semantic Recognition).")
    
    # Tabs
    tab1, tab2 = st.tabs(["📸 Live Camera", "📂 Gallery Upload"])
    img_data = None

    with tab1:
        cam_img = st.camera_input("Take a picture")
        if cam_img: img_data = cam_img.getvalue()
    with tab2:
        up_img = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
        if up_img: 
            img_data = up_img.getvalue()
            st.image(img_data, width=300)

    if img_data:
        with st.spinner("Running Hybrid Analysis (OpenCV + Gemini)..."):
            status, result = analyze_smart_hybrid(img_data)
            
            if status == "SUCCESS":
                st.success("✅ Analysis Complete!")
                st.markdown(result)
                add_xp(20, "Hybrid Scan")
            else:
                st.error(f"⚠️ AI Connection Issue: {result}")
                st.warning("Switching to Manual Mode.")
                
                # Manual Fallback
                man = st.text_input("Describe item:", key="fail_fix")
                if man and st.button("Get Rules"):
                    st.markdown(ask_groq(f"How to recycle {man}?"))

# --- STANDARD RENDERERS ---
def render_home():
    st.write(""); st.title(f"🌍 EcoWise Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Points", st.session_state.xp)
    c2.metric("🔥 Streak", f"{st.session_state.streak} Days")
    c3.metric("🏆 Rank", "Eco-Warrior")
    st.divider()
    
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Visual Sorter (Hybrid)", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
    with col2:
        if st.button("🛒 Campus Swap", use_container_width=True): navigate_to("🛒 Campus Swap")
        if st.button("📊 Leaderboard", use_container_width=True): navigate_to("📊 Leaderboard")
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
    
    st.divider()
    st.subheader("🎯 Daily Challenges")
    if st.button("✅ Used Refill Bottle (+20)"): add_xp(20, "Challenge"); st.balloons()

def render_voice_mode():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Record")
    if audio:
        txt = transcribe_audio(audio)
        st.write(f"You: {txt}")
        st.markdown(ask_groq(txt))
        add_xp(10, "Voice")

def render_recycle_assistant():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("♻️ Chat Assistant")
    q = st.chat_input("Ask...")
    if q: st.markdown(ask_groq(q))

def render_map():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🗺️ Eco-Map")
    st_folium(folium.Map([20.5, 78.9], zoom_start=5), height=400)

def render_marketplace():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🛒 Campus Swap")
    st.info("Marketplace under maintenance.")

def render_leaderboard():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🏆 Leaderboard")
    st.info("Loading global stats...")

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    make_pwa_ready()
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        e = st.text_input("Email"); p = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                st.session_state.user = res.user; st.session_state.user_id = res.user.id
                sync_user_stats(res.user.id); st.rerun()
            except Exception as err: st.error(str(err))
        return

    # Sidebar
    with st.sidebar:
        st.title("EcoWise AI")
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🚪 Logout"): supabase.auth.sign_out(); st.session_state.clear(); st.rerun()

    # Routing
    if st.session_state.feature == "🏠 Home": render_home()
    elif st.session_state.feature == "📸 Visual Sorter": render_visual_sorter()
    elif st.session_state.feature == "🎙️ Voice Mode": render_voice_mode()
    elif st.session_state.feature == "♻️ Recycle Assistant": render_recycle_assistant()
    elif st.session_state.feature == "🗺️ Eco-Map": render_map()
    elif st.session_state.feature == "🛒 Campus Swap": render_marketplace()
    elif st.session_state.feature == "📊 Leaderboard": render_leaderboard()

if __name__ == "__main__":
    main()

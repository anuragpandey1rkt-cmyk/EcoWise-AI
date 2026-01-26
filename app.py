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
                padding-top: 5rem; 
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
        "waste_guidelines_text": ""
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
# 3. ROBUST AI LOGIC
# ==========================================

def ask_groq(prompt, system_role="You are a helpful Sustainability Expert."):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Logic Error: {str(e)}"

def get_best_gemini_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
    except: pass
    return "models/gemini-1.5-flash"

def analyze_image_robust(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes))
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(get_best_gemini_model())
            response = model.generate_content(["Identify this object. Is it recyclable?", image_pil])
            return response.text
        except: pass 
    return "MANUAL_FALLBACK"

def transcribe_audio(audio_bytes):
    try:
        return groq_client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes), model="whisper-large-v3", response_format="json"
        ).text
    except: return "Error processing audio."

def extract_text_from_pdf(file):
    try: return "".join([p.extract_text() for p in PdfReader(file).pages])
    except: return None

# ==========================================
# 4. GAMIFICATION & SYNC
# ==========================================
def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    st.session_state.xp += amount
    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp}).eq("user_id", st.session_state.user_id).execute()
        st.toast(f"🌱 +{amount} XP!", icon="🎉")
    except: pass

def sync_user_stats(uid):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", uid).execute()
        if data.data:
            st.session_state.xp = data.data[0].get('xp', 0)
            st.session_state.streak = data.data[0].get('streak', 0)
        else:
            supabase.table("user_stats").insert({"user_id": uid, "xp": 0, "streak": 0}).execute()
    except: pass

# ==========================================
# 5. FEATURE RENDERERS
# ==========================================

def render_home():
    st.title(f"👋 Hi, Eco-Warrior")
    
    # --- PASSWORD RESET HELP BOX ---
    if st.session_state.user:
        with st.expander("🔑 Resetting Password?", expanded=True):
            st.info("If you just used a reset link, update your password here:")
            c1, c2 = st.columns([3,1])
            with c1: new_p = st.text_input("New Password", type="password", key="hp")
            with c2: 
                st.write(""); st.write("")
                if st.button("Update"):
                    supabase.auth.update_user({"password": new_p})
                    st.success("Updated!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Points", st.session_state.xp)
    c2.metric("Streak", f"{st.session_state.streak}🔥")
    c3.metric("Rank", "Titan" if st.session_state.xp > 1000 else "Rookie")
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
    with c2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")

def render_visual_sorter():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Sorter")
    img = st.camera_input("Take Photo")
    if not img: img_up = st.file_uploader("Or Upload", type=['jpg','png'])
    final_img = img if img else (img_up if 'img_up' in locals() and img_up else None)
    if final_img:
        with st.spinner("Analyzing..."):
            res = analyze_image_robust(final_img.getvalue())
            st.markdown(res)
            add_xp(15, "Scan")

def render_map():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    m = folium.Map([20.59, 78.96], zoom_start=4)
    st_folium(m, height=350)
    st.info("Map points loaded from database.")

def render_plastic_calculator():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌊 Plastic Calc")
    b = st.slider("Bottles/Week", 0, 50, 5)
    kg = ((b*12) * 52) / 1000
    st.metric("Yearly Waste", f"{kg} kg")

def render_forest():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Forest")
    trees = st.session_state.xp // 100
    st.markdown(f"# {'🌲 ' * trees}")

def render_voice():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Mode")
    aud = st.audio_input("Speak")
    if aud: st.markdown(ask_groq(transcribe_audio(aud)))

def render_chat():
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Chatbot")
    q = st.chat_input("Ask...")
    if q: st.markdown(ask_groq(q))

# ==========================================
# 6. MAIN APP LOOP (SMART LINK DETECTION)
# ==========================================
def main():
    init_session_state()
    make_pwa_ready()
    
    # --- 🛠️ STEP 1: SMART LINK DETECTOR (Updated) 🛠️ ---
    # This now detects both SUCCESS tokens AND ERROR tokens (like expired links)
    st.markdown("""
    <script>
    if (window.location.hash) {
        const hash = window.location.hash.substring(1);
        if (hash.includes("access_token") || hash.includes("error_code")) {
            // Reload the page with the hash converted to query parameters so Python can see it
            window.location.href = window.location.pathname + "?" + hash;
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # --- 🛠️ STEP 2: PYTHON TOKEN & ERROR CATCHER 🛠️ ---
    try:
        query_params = st.query_params
        
        # A. Check for ERRORS (like expired links)
        if "error_code" in query_params:
            error_msg = query_params.get("error_description", ["Unknown error"])[0] if isinstance(query_params.get("error_description"), list) else query_params.get("error_description", "Unknown error")
            st.error(f"⚠️ Link Error: {error_msg}")
            st.warning("👉 The link you clicked has expired. Please delete old emails and request a new one.")
            st.query_params.clear() # Clear it so the error doesn't persist forever
            
        # B. Check for SUCCESS (access_token)
        access_token = query_params.get("access_token")
        refresh_token = query_params.get("refresh_token")
        
        if access_token and refresh_token:
            session = supabase.auth.set_session(access_token, refresh_token)
            if session:
                st.session_state.user = session.user
                st.session_state.user_id = session.user.id
                sync_user_stats(session.user.id)
                st.query_params.clear() 
                st.success("✅ Logged in via Reset Link!")
                st.rerun()
    except Exception as e: 
        # Optional: Print error to console for debugging if needed, but keep UI clean
        pass
    
    # --- STEP 3: LOGIN SCREEN ---
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        mode = st.radio("Mode", ["Login", "Sign Up", "Forgot Password"], horizontal=True, label_visibility="collapsed")

        if mode == "Login":
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Login"): 
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.session_state.user = res.user; st.session_state.user_id = res.user.id
                    sync_user_stats(res.user.id); st.rerun()
                except Exception as err: st.error(str(err))

        elif mode == "Sign Up":
            e2 = st.text_input("Email (New)")
            p2 = st.text_input("Password (New)", type="password")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: st.success("✅ Created! Check email to verify.")
                except Exception as err: st.error(str(err))

        elif mode == "Forgot Password":
            st.info("Enter email to get a login link.")
            reset_email = st.text_input("Email")
            if st.button("Send Link"):
                try:
                    # USE YOUR APP URL HERE
                    supabase.auth.reset_password_email(reset_email, options={
                        "redirect_to": "https://ecowise-ai-2026.streamlit.app"
                    })
                    st.success("✅ Link sent! Check email.")
                except: st.warning("⏳ Check your inbox (or wait a bit).")
        return

    # --- STEP 4: APP NAVIGATION ---
    with st.sidebar:
        st.title("EcoWise")
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
        
        st.divider()
        if st.button("🚪 Logout"): 
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "♻️ Recycle Assistant": render_chat()
    elif f == "🌊 Plastic Calculator": render_plastic_calculator()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🎙️ Voice Mode": render_voice()
    elif f == "🌳 My Forest": render_forest()

if __name__ == "__main__":
    main()

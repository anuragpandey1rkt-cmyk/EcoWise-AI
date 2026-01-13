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
# 1. HELPER FUNCTIONS (DEFINED FIRST TO FIX ERRORS)
# ==========================================

def make_pwa_ready():
    """Styles the app to look like a mobile app"""
    st.markdown("""
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <style>
            /* Clean up the UI but don't hide the header completely */
            footer {visibility: hidden;}
            
            /* Mobile-friendly container */
            div.block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
            }
            
            /* Big, touch-friendly buttons */
            div.stButton > button {
                width: 100%;
                border-radius: 10px;
                height: 3rem;
                font-weight: 600;
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

# Load Secrets safely
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
# 3. ROBUST AI LOGIC (FIXES 404 ERRORS)
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
    return "models/gemini-1.5-flash" # Default fallback

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
# 4. GAMIFICATION & SYNC
# ==========================================
def add_xp(amount, activity):
    if not st.session_state.user_id: return
    st.session_state.xp += amount
    today = str(datetime.date.today())
    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp}).eq("user_id", st.session_state.user_id).execute()
        st.toast(f"+{amount} XP!", icon="🎉")
        if st.session_state.last_action_date != today:
            st.session_state.streak += 1
            st.session_state.last_action_date = today
            supabase.table("user_stats").update({"streak": st.session_state.streak, "last_study_date": today}).eq("user_id", st.session_state.user_id).execute()
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
# 5. FEATURE SCREENS
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
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
    with c2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")

def render_visual_sorter():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Sorter")
    img = st.camera_input("Take Photo")
    if not img: img_up = st.file_uploader("Or Upload", type=['jpg','png'])
    
    final_img = img if img else (img_up if 'img_up' in locals() and img_up else None)

    if final_img:
        with st.spinner("Analyzing..."):
            res = analyze_image_robust(final_img.getvalue())
            if res == "MANUAL_FALLBACK":
                st.warning("⚠️ AI busy. Type item name:")
                man = st.text_input("Item Name")
                if man and st.button("Check"):
                    st.markdown(ask_groq(f"Recycle {man}"))
            else:
                st.success("✅ Identified!")
                st.markdown(res)
                add_xp(15, "Scan")

def render_map():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    st.info("Click map to pin a spot!")
    
    m = folium.Map([20.59, 78.96], zoom_start=4)
    pts = supabase.table("map_points").select("*").execute().data
    for p in pts: folium.Marker([p['latitude'], p['longitude']], popup=p['name']).add_to(m)
    
    data = st_folium(m, height=350)
    lat, lon = 20.59, 78.96
    if data and data.get("last_clicked"):
        lat, lon = data["last_clicked"]["lat"], data["last_clicked"]["lng"]
        st.write(f"📍 Selected: {lat:.4f}, {lon:.4f}")
    
    with st.form("pin"):
        n = st.text_input("Name"); t = st.selectbox("Type", ["Bin", "E-Waste"])
        if st.form_submit_button("Pin"):
            supabase.table("map_points").insert({"user_id": st.session_state.user_id, "name": n, "latitude": lat, "longitude": lon, "type": t}).execute()
            st.success("Pinned!"); st.rerun()

def render_plastic_calculator():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌊 Plastic Calc")
    b = st.slider("Bottles/Week", 0, 50, 5)
    bg = st.slider("Bags/Week", 0, 50, 5)
    kg = ((b*12 + bg*5) * 52) / 1000
    st.metric("Yearly Waste", f"{kg} kg")
    if st.button("Reduction Plan"):
        st.markdown(ask_groq(f"How to reduce {kg}kg plastic waste?"))
        add_xp(20, "Audit")

def render_upcycling():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎨 Upcycling")
    item = st.text_input("I have an old...")
    if item and st.button("Get Ideas"):
        st.markdown(ask_groq(f"DIY ideas for {item}"))
        add_xp(20, "Upcycle")

def render_menu():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🥗 Eco-Menu")
    c = st.selectbox("Cuisine", ["Indian", "Italian", "Mexican"])
    if st.button("Plan"):
        st.markdown(ask_groq(f"Low-carbon {c} meal plan"))
        add_xp(20, "Menu")

def render_forest():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Forest")
    trees = st.session_state.xp // 100
    st.metric("Trees", trees)
    st.markdown(f"# {'🌲 ' * trees}")

def render_voice():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Mode")
    aud = st.audio_input("Speak")
    if aud:
        txt = transcribe_audio(aud)
        st.write(f"You: {txt}")
        st.markdown(ask_groq(txt))

def render_chat():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Chatbot")
    up = st.file_uploader("Rules PDF", type=['pdf'])
    if up: st.session_state.waste_guidelines_text = extract_text_from_pdf(up)
    q = st.chat_input("Ask...")
    if q: st.markdown(ask_groq(q + (st.session_state.waste_guidelines_text or "")))

def render_mistake():
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Fixer")
    m = st.text_input("Threw what?")
    if st.button("Check"): st.markdown(ask_groq(f"I threw {m} in trash. Bad?"))

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
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")
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

if __name__ == "__main__":
    main()

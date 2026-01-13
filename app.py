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
import cv2 # OpenCV
import numpy as np

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
except FileNotFoundError:
    st.error("Secrets not found. Please set up .streamlit/secrets.toml")
    st.stop()

# Initialize Clients
@st.cache_resource
def init_clients():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    return supa, groq

supabase, groq_client = init_clients()

# ==========================================
# 2. SESSION STATE
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
# 3. ADVANCED OPENCV ENGINE (LOCAL AI)
# ==========================================

def get_dominant_color(image_array):
    """
    Scans the image to find the main color (Green, Brown, Blue, etc.)
    Used to guess material (Cardboard, Glass, Plastic).
    """
    # Resize for speed
    img = cv2.resize(image_array, (50, 50))
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0) # B, G, R
    
    blue, green, red = avg_color
    
    # Simple Color Heuristics
    if red > 150 and green > 150 and blue > 150: return "White/Light (Paper/Plastic)"
    if red > 100 and green > 100 and blue < 100: return "Yellow (Organic/Peel)"
    if green > red and green > blue: return "Green (Glass/Organic)"
    if blue > red and blue > green: return "Blue (Plastic/Wrapper)"
    if red > 100 and green < 80 and blue < 80: return "Red (Hazardous/Plastic)"
    if red > 100 and green > 80 and blue < 60: return "Brown (Cardboard/Organic)"
    return "Dark/Mixed"

def analyze_with_opencv(image_bytes):
    """
    Performs Computer Vision Analysis:
    1. Edge Detection (Canny)
    2. Contour Analysis (Shape Detection)
    3. Color Analysis (Material Guess)
    """
    # Convert bytes to numpy array
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 1. Image Quality Check (Blurry?)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Shape Detection (Aspect Ratio)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_guess = "Unknown"
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(h) / w
        
        if aspect_ratio > 1.5: shape_guess = "Tall/Cylindrical (Bottle/Can)"
        elif aspect_ratio < 0.8: shape_guess = "Wide/Flat (Box/Tray)"
        else: shape_guess = "Square/Round (Container/Fruit)"
        
    # 3. Color Analysis
    color_desc = get_dominant_color(image)
    
    # 4. Generate Report
    report = (
        f"**Computer Vision Analysis:**\n"
        f"- **Shape:** {shape_guess}\n"
        f"- **Dominant Color:** {color_desc}\n"
        f"- **Image Sharpness:** {int(laplacian_var)}\n\n"
        f"Based on visual features, please confirm the item below to get recycling rules."
    )
    return report, shape_guess, color_desc

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
            st.session_state.last_action_date = stats.get('last_study_date')
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
    st.header("📸 OpenCV Waste Sorter")
    st.info("Using Computer Vision (Edge & Color Detection) to analyze items locally.")
    
    # Tabs for Camera or Upload
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
        with st.spinner("Processing with OpenCV..."):
            # 1. RUN OPENCV LOCAL ANALYSIS
            report, shape, color = analyze_with_opencv(img_data)
            st.success("✅ OpenCV Analysis Complete!")
            
            # 2. SHOW TECHNICAL DATA (Looks impressive)
            c1, c2 = st.columns(2)
            c1.info(f"📐 Shape: {shape}")
            c2.info(f"🎨 Color: {color}")
            
            # 3. CONFIRMATION (Since OpenCV can't know 'Brand Name')
            st.write("Confirm item to get recycling rules:")
            
            # Pre-fill guess based on OpenCV data
            guess = ""
            if "Bottle" in shape: guess = "Plastic Bottle"
            elif "Box" in shape: guess = "Cardboard Box"
            elif "Brown" in color: guess = "Cardboard"
            elif "Green" in color: guess = "Glass Bottle"
            
            item_name = st.text_input("Item Name:", value=guess)
            
            if st.button("♻️ Get Disposal Instructions"):
                with st.spinner("Consulting Database..."):
                    # Use Groq for the logic part
                    prompt = f"How do I recycle a '{item_name}'? It is {shape} and {color}. Be strict."
                    advice = ask_groq(prompt)
                    st.markdown(f"### 📋 Instructions for {item_name}")
                    st.markdown(advice)
                    add_xp(20, "OpenCV Scan")

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
        if st.button("📸 Visual Sorter (OpenCV)", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
    with col2:
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
    
    st.divider()
    st.subheader("🎯 Daily Challenges")
    if st.button("✅ Refused Plastic (+20)"): add_xp(20, "Challenge"); st.balloons()

# --- OTHER RENDERERS (Standard) ---
def render_voice_mode():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Record")
    if audio:
        txt = transcribe_audio(audio)
        st.write(f"You: {txt}")
        st.write(ask_groq(txt))
        add_xp(10, "Voice")

def render_recycle_assistant():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("♻️ Chat Assistant")
    q = st.chat_input("Ask...")
    if q: st.write(ask_groq(q))

def render_map():
    st.write(""); st.button("⬅️ Back", on_click=navigate_to, args=("🏠 Home",))
    st.header("🗺️ Eco-Map")
    st_folium(folium.Map([20.5, 78.9], zoom_start=5), height=400)

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

if __name__ == "__main__":
    main()

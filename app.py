import streamlit as st
import time
import base64
import pandas as pd
import random
import requests
import io
import os
from PIL import Image
from supabase import create_client
from groq import Groq
from PyPDF2 import PdfReader
import folium
from streamlit_folium import st_folium
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

# Custom CSS for Mobile (PWA)
st.markdown("""
    <style>
        .block-container { padding-top: 3rem; padding-bottom: 5rem; }
        div.stButton > button { width: 100%; border-radius: 8px; height: 3rem; }
    </style>
""", unsafe_allow_html=True)

# --- SECRETS MANAGER ---
# We use .get() so the app doesn't crash if a key is missing
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
except FileNotFoundError:
    st.error("CRITICAL ERROR: Secrets file not found.")
    st.stop()

# --- INIT CLIENTS ---
@st.cache_resource
def init_clients():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    return supa, groq

supabase, groq_client = init_clients()

# ==========================================
# 2. SESSION STATE
# ==========================================
if 'user' not in st.session_state:
    st.session_state.user = None
if 'feature' not in st.session_state:
    st.session_state.feature = "🏠 Home"
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'waste_guidelines' not in st.session_state:
    st.session_state.waste_guidelines = ""

def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 3. ROBUST "MULTI-ENGINE" AI PIPELINE
# ==========================================

def ask_groq(prompt, context=""):
    """
    The 'Brain' of the operation. Uses Llama 3 via Groq for logic.
    """
    system_prompt = "You are an expert in Waste Management and Sustainability. Be brief, practical, and motivating."
    if context:
        system_prompt += f"\n\nCONTEXT FROM USER DOCS: {context[:5000]}"
        
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Logic Error: {str(e)}"

def analyze_image_pipeline(image_bytes):
    """
    THE POWERFUL ROBUST CODE:
    1. Try Google Gemini (Best Quality)
    2. Try Hugging Face BLIP (Best Availability - Acts like OpenCV)
    3. Fallback to Manual
    """
    # Convert bytes to PIL Image
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
    except:
        return "ERROR: Invalid Image Format"

    # --- ENGINE 1: GOOGLE GEMINI ---
    if GEMINI_API_KEY:
        models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro-vision']
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    "Identify this item exactly. Is it recyclable? How do I dispose of it? Be brief.", 
                    image_pil
                ])
                if response.text:
                    return f"**✅ Verified by Gemini ({model_name}):**\n\n{response.text}"
            except:
                continue # Silently try next model

    # --- ENGINE 2: HUGGING FACE (The "OpenCV" Alternative) ---
    # This uses a captioning model to "see" the image textually
    if HF_TOKEN:
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        try:
            response = requests.post(API_URL, headers=headers, data=image_bytes)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and 'generated_text' in data[0]:
                    # The AI "saw" the image
                    detected_item = data[0]['generated_text']
                    
                    # Now ask Groq for the logic
                    advice = ask_groq(f"I have an item that looks like '{detected_item}'. Is it recyclable? How do I dispose of it?")
                    return f"**📷 Detected:** '{detected_item.title()}'\n\n{advice}"
        except:
            pass # Fall through to manual

    # --- ENGINE 3: FAILURE SIGNAL ---
    return "MANUAL_FALLBACK"

# --- HELPER FUNCTIONS ---
def add_xp(amount, activity):
    if st.session_state.user:
        st.session_state.xp += amount
        # In a real app, you would update Supabase here
        st.toast(f"+{amount} XP: {activity}", icon="🎉")

def transcribe_audio(audio_bytes):
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
            language="en"
        )
        return transcription.text
    except:
        return "Error hearing audio."

# ==========================================
# 4. FEATURE SCREENS
# ==========================================

def render_home():
    st.write("")
    st.title("🌍 EcoWise Dashboard")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 XP", st.session_state.xp)
    c2.metric("🔥 Streak", "3 Days")
    c3.metric("🌱 Rank", "Novice")
    
    st.divider()
    
    # Navigation Grid
    st.subheader("🚀 Tools")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📸 Visual Sorter", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycle Guide", use_container_width=True): navigate_to("♻️ Recycle Assistant")
    with c2:
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
        if st.button("🛒 Campus Swap", use_container_width=True): navigate_to("🛒 Campus Swap")
        if st.button("❌ Mistake Fixer", use_container_width=True): navigate_to("❌ Mistake Explainer")

def render_visual_sorter():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 Smart Vision Sorter")
    st.info("Uses Multi-Engine AI to identify trash.")

    # Two input methods
    tab1, tab2 = st.tabs(["📷 Camera", "📂 Upload"])
    img_data = None

    with tab1:
        cam = st.camera_input("Snap Photo")
        if cam: img_data = cam.getvalue()
    with tab2:
        up = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if up: img_data = up.getvalue()

    if img_data:
        with st.spinner("🔍 Scanning Object (Trying Google -> Hugging Face)..."):
            result = analyze_image_pipeline(img_data)
            
            if result == "MANUAL_FALLBACK":
                st.warning("⚠️ AI Vision is busy/unreachable.")
                st.write("Please identify the item manually:")
                item = st.text_input("Item Name (e.g. Plastic Bottle)")
                if item and st.button("Get Instructions"):
                    res = ask_groq(f"How do I recycle {item}?")
                    st.success("✅ Instructions Found")
                    st.markdown(res)
                    add_xp(20, "Manual ID")
            else:
                st.success("✅ Object Identified!")
                st.markdown(result)
                add_xp(20, "AI Scan")

def render_voice_mode():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    
    audio = st.audio_input("Ask a question...")
    if audio:
        with st.spinner("Listening..."):
            text = transcribe_audio(audio)
            st.write(f"**You:** {text}")
            response = ask_groq(text)
            st.markdown(f"**EcoWise:** {response}")
            add_xp(10, "Voice Chat")

def render_recycle_assistant():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Recycle Assistant")
    
    with st.expander("📂 Upload Local Rules (PDF)"):
        pdf = st.file_uploader("Upload PDF", type='pdf')
        if pdf:
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages: text += page.extract_text()
            st.session_state.waste_guidelines = text
            st.success("Rules Indexed!")

    q = st.chat_input("Ask about recycling...")
    if q:
        res = ask_groq(q, st.session_state.waste_guidelines)
        st.write(res)
        add_xp(5, "Chat")

def render_map():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    st.caption("Find local bins and centers.")
    
    # Default Map (India Center)
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Fetch points from DB
    try:
        pts = supabase.table("map_points").select("*").execute().data
        for p in pts:
            folium.Marker(
                [p['latitude'], p['longitude']], 
                popup=p['name'], 
                icon=folium.Icon(color="green", icon="trash")
            ).add_to(m)
    except:
        st.caption("No map points found in DB.")

    st_folium(m, height=400, use_container_width=True)
    
    with st.expander("📍 Add New Spot"):
        with st.form("add_map"):
            name = st.text_input("Location Name")
            lat = st.number_input("Latitude", value=20.0)
            lon = st.number_input("Longitude", value=78.0)
            if st.form_submit_button("Add Pin"):
                try:
                    supabase.table("map_points").insert({
                        "user_id": st.session_state.user.id,
                        "name": name, "latitude": lat, "longitude": lon, "type": "recycle"
                    }).execute()
                    st.success("Added!")
                    st.rerun()
                except:
                    st.error("Login required to add points.")

def render_marketplace():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🛒 Campus Swap")
    
    tab1, tab2 = st.tabs(["Buy", "Sell"])
    
    with tab1:
        try:
            items = supabase.table("marketplace_items").select("*").execute().data
            if items:
                for i in items:
                    with st.container(border=True):
                        st.subheader(i['item_name'])
                        st.write(f"💰 {i['price']}")
                        st.caption(f"📞 {i['contact_info']}")
            else:
                st.info("No items listed yet.")
        except:
            st.error("Database connection error.")

    with tab2:
        with st.form("sell_form"):
            name = st.text_input("Item Name")
            price = st.text_input("Price")
            contact = st.text_input("Contact Info")
            if st.form_submit_button("List Item"):
                try:
                    supabase.table("marketplace_items").insert({
                        "user_id": st.session_state.user.id,
                        "item_name": name, "price": price, "contact_info": contact
                    }).execute()
                    st.success("Listed!")
                    st.rerun()
                except:
                    st.error("Login required.")

def render_mistake_explainer():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    
    col1, col2 = st.columns(2)
    with col1:
        item = st.text_input("I threw away...")
    with col2:
        bin_type = st.selectbox("Into the...", ["Trash Bin", "Recycle Bin", "Compost", "Toilet"])
        
    if st.button("Analyze Mistake"):
        res = ask_groq(f"I threw {item} into the {bin_type}. Explain the environmental impact and the correct way.")
        st.error("📉 Impact Analysis")
        st.write(res)
        add_xp(15, "Learned from Mistake")

# ==========================================
# 5. MAIN AUTH LOOP
# ==========================================
def main():
    make_pwa_ready()
    
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
        
        with tab2:
            e2 = st.text_input("Sign Up Email")
            p2 = st.text_input("Sign Up Password", type="password")
            if st.button("Create Account"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user:
                        st.success("Account created! You can now login.")
                        # Init user stats
                        supabase.table("user_stats").insert({"user_id": res.user.id, "xp": 0}).execute()
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.title("EcoWise Menu")
        st.write(f"User: {st.session_state.user.email}")
        st.divider()
        if st.button("🚪 Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    # Routing
    if st.session_state.feature == "🏠 Home": render_home()
    elif st.session_state.feature == "📸 Visual Sorter": render_visual_sorter()
    elif st.session_state.feature == "🎙️ Voice Mode": render_voice_mode()
    elif st.session_state.feature == "♻️ Recycle Assistant": render_recycle_assistant()
    elif st.session_state.feature == "🗺️ Eco-Map": render_map()
    elif st.session_state.feature == "🛒 Campus Swap": render_marketplace()
    elif st.session_state.feature == "❌ Mistake Explainer": render_mistake_explainer()

if __name__ == "__main__":
    main()

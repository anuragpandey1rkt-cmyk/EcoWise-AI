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
    # Optional Keys (Code works even if one is missing)
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except FileNotFoundError:
    st.error("Secrets not found. Please set up .streamlit/secrets.toml")
    st.stop()

# Initialize Clients
@st.cache_resource
def init_clients():
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    
    # Configure Gemini if key exists
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        
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
# 3. TRIPLE-LAYER AI ENGINE
# ==========================================

def ask_groq(prompt, system_role="You are a helpful Sustainability Expert."):
    """Uses Groq (Llama 3) for logic and text"""
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

def analyze_image_final(image_bytes):
    """
    TRIPLE LAYER FALLBACK SYSTEM:
    1. Try Google Gemini (Cycle through 4 model names)
    2. Try Hugging Face (BLIP-Base - Faster model)
    3. Return MANUAL_FALLBACK signal
    """
    image_pil = Image.open(io.BytesIO(image_bytes))

    # --- LAYER 1: GOOGLE GEMINI (Brute Force) ---
    if GEMINI_API_KEY:
        # List of models to try in order. This solves the "404" issue.
        gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.5-pro-001"
        ]
        
        for model_name in gemini_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([
                    "Identify this object exactly. Is it recyclable, compostable, or trash? Be brief and give strict disposal instructions.", 
                    image_pil
                ])
                return response.text # Success! Return immediately.
            except Exception:
                continue # Try next model name
                
    # --- LAYER 2: HUGGING FACE (Lighter Model) ---
    if HF_TOKEN:
        try:
            # We use 'blip-image-captioning-base' (Smaller/Faster than Large)
            API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            
            response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=8)
            
            if response.status_code == 200:
                prediction = response.json()
                if isinstance(prediction, list) and 'generated_text' in prediction[0]:
                    item_name = prediction[0]['generated_text']
                    
                    # Pass the name to Groq for the recycling advice
                    advice = ask_groq(f"How do I recycle '{item_name}'? Be strict.")
                    return f"**Detected:** {item_name.title()}\n\n{advice}"
        except Exception as e:
            print(f"HF Failed: {e}")

    # --- LAYER 3: SIGNAL TO UI ---
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
    except Exception as e:
        return None

# --- DATABASE SYNC ---
def sync_user_stats(user_id):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", user_id).execute()
        if data.data:
            stats = data.data[0]
            st.session_state.xp = stats.get('xp', 0)
            st.session_state.streak = stats.get('streak', 0)
        else:
            supabase.table("user_stats").insert({"user_id": user_id, "xp": 0, "streak": 0}).execute()
    except Exception as e:
        print(f"Sync Error: {e}")

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
    except Exception as e:
        st.error(f"Sync Error: {e}")

# ==========================================
# 4. RENDERERS
# ==========================================

def render_home():
    st.write("") 
    st.title(f"🌍 EcoWise Dashboard")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Points", st.session_state.xp)
    c2.metric("🔥 Streak", f"{st.session_state.streak} Days")
    c3.metric("🏆 Rank", "Eco-Warrior" if st.session_state.xp > 500 else "Rookie")
    
    st.divider()
    
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Visual Sorter", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("❌ Mistake Explainer", use_container_width=True): navigate_to("❌ Mistake Explainer")
    with col2:
        if st.button("🛒 Campus Swap", use_container_width=True): navigate_to("🛒 Campus Swap")
        if st.button("📊 Leaderboard", use_container_width=True): navigate_to("📊 Leaderboard")
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
        if st.button("👣 Carbon Tracker", use_container_width=True): navigate_to("👣 Carbon Tracker")

    st.divider()
    st.subheader("🎯 Daily Green Challenges")
    
    today = str(datetime.date.today())
    if st.session_state.last_challenge_date != today:
        possible = ["Use refillable bottle", "Recycle 3 items", "Plant-based meal", "Unplug electronics", "Pick up litter"]
        st.session_state.daily_challenges = random.sample(possible, 3)
        st.session_state.last_challenge_date = today

    for i, task in enumerate(st.session_state.daily_challenges):
        c_a, c_b = st.columns([4, 1])
        c_a.write(f"✅ **{task}**")
        if c_b.button(f"Done (+20)", key=f"d_{i}"):
            add_xp(20, f"Challenge: {task}")
            st.balloons()

def render_visual_sorter():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Visual Waste Sorter")
    st.info("Identify trash instantly using Smart AI.")
    
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
        with st.spinner("Analyzing (Trying Google -> Hugging Face)..."):
            res = analyze_image_final(img_data)
            
            # --- INTELLIGENT FALLBACK UI ---
            if res == "MANUAL_FALLBACK":
                st.warning("⚠️ Vision AI is currently busy or unreachable.")
                st.info("💡 Don't worry! You can still get points by identifying the item.")
                
                # Manual Input - Keeps the flow going!
                man = st.text_input("What item is this?", key="fallback_input", placeholder="e.g. Plastic Bottle")
                
                if man and st.button("Get Instructions"):
                    with st.spinner("Checking Eco-Database..."):
                        advice = ask_groq(f"How do I recycle a {man}? Be strict.")
                        st.success(f"Instructions for: {man}")
                        st.markdown(advice)
                        add_xp(15, "Manual Scan")
            else:
                st.success("✅ Analysis Complete!")
                st.markdown(res)
                add_xp(15, "Visual Scan")

def render_voice_mode():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Record")
    if audio:
        with st.spinner("Listening..."):
            txt = transcribe_audio(audio)
            st.write(f"**You said:** {txt}")
            res = ask_groq(txt)
            st.markdown(f"**AI:** {res}")
            add_xp(10, "Voice Query")

def render_recycle_assistant():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Smart Recycle Assistant")
    with st.expander("📂 Upload Municipal Rules (PDF)"):
        f = st.file_uploader("Upload PDF", type=['pdf'])
        if f: 
            st.session_state.waste_guidelines_text = extract_text_from_pdf(f)
            st.success("Rules Loaded!")
    q = st.chat_input("Ask about recycling...")
    if q:
        role = "Waste Expert."
        if st.session_state.waste_guidelines_text: role += f"\nData:\n{st.session_state.waste_guidelines_text[:10000]}"
        res = ask_groq(q, role)
        st.write(res); add_xp(5, "Query")

def render_mistake_explainer():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    m = st.text_input("I wrongly disposed of...")
    b = st.selectbox("Into...", ["Recycle Bin", "Compost", "Trash"])
    if st.button("Explain Impact"):
        st.markdown(ask_groq(f"I put {m} into {b}. Explain environmental consequence."))
        add_xp(10, "Learning")

def render_map():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Eco-Map")
    with st.expander("📍 Add Spot"):
        with st.form("map"):
            n = st.text_input("Name"); lat = st.number_input("Lat", 28.6); lon = st.number_input("Lon", 77.2)
            t = st.selectbox("Type", ["Recycle", "Donation"])
            if st.form_submit_button("Add"):
                supabase.table("map_points").insert({"user_id": st.session_state.user_id, "name": n, "latitude": lat, "longitude": lon, "type": t}).execute()
                st.success("Added!"); st.rerun()
    pts = supabase.table("map_points").select("*").execute().data
    start = [pts[0]['latitude'], pts[0]['longitude']] if pts else [20.5, 78.9]
    m = folium.Map(start, zoom_start=10)
    for p in pts: folium.Marker([p['latitude'], p['longitude']], popup=p['name'], icon=folium.Icon(color="green")).add_to(m)
    st_folium(m, height=400)

def render_marketplace():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🛒 Campus Swap")
    t1, t2 = st.tabs(["Browse", "Sell"])
    with t1:
        items = supabase.table("marketplace_items").select("*").execute().data
        if items:
            for i in items: st.info(f"**{i['item_name']}** | {i['price']} | {i['contact_info']}")
        else: st.info("No items.")
    with t2:
        with st.form("sell"):
            n = st.text_input("Item"); p = st.text_input("Price"); c = st.text_input("Contact")
            if st.form_submit_button("List"):
                supabase.table("marketplace_items").insert({"user_id": st.session_state.user_id, "item_name": n, "price": p, "contact_info": c}).execute()
                st.success("Listed!"); st.rerun()

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

def render_carbon_tracker():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("👣 Carbon Tracker")
    t = st.selectbox("Transport", ["Walk", "Car", "Bus"])
    if st.button("Log"): add_xp(20, f"Transport: {t}"); st.success("Logged!")

def main():
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
            e2 = st.text_input("Email (Sign Up)")
            p2 = st.text_input("Password (Sign Up)", type="password")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: 
                        supabase.table("user_stats").insert({"user_id": res.user.id}).execute()
                        st.success("Created! Login now.")
                except Exception as err: st.error(str(err))
        return

    with st.sidebar:
        st.title("EcoWise AI")
        st.write(f"👤 {st.session_state.user.email}")
        st.divider()
        if st.button("🏠 Home", use_container_width=True): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
        if st.button("🛒 Campus Swap", use_container_width=True): navigate_to("🛒 Campus Swap")
        if st.button("📊 Leaderboard", use_container_width=True): navigate_to("📊 Leaderboard")
        st.divider()
        if st.button("🚪 Logout"): 
            supabase.auth.sign_out()
            st.session_state.clear()
            st.rerun()

    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "🎙️ Voice Mode": render_voice_mode()
    elif f == "♻️ Recycle Assistant": render_recycle_assistant()
    elif f == "❌ Mistake Explainer": render_mistake_explainer()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🛒 Campus Swap": render_marketplace()
    elif f == "📊 Leaderboard": render_leaderboard()
    elif f == "👣 Carbon Tracker": render_carbon_tracker()

if __name__ == "__main__":
    main()

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
# 1. APP CONFIGURATION & MOBILE UI
# ==========================================
st.set_page_config(
    page_title="EcoWise AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_mobile_styles():
    st.markdown("""
        <style>
            /* Hide Streamlit Elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Mobile App Container */
            div.block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
                max-width: 800px; /* Force mobile width feeling on desktop */
                margin: auto;
            }
            
            /* App-like Buttons */
            div.stButton > button {
                width: 100%;
                border-radius: 12px;
                height: 3.5rem;
                font-weight: bold;
                border: none;
                box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                transition: all 0.2s;
            }
            div.stButton > button:active {
                transform: scale(0.98);
            }
            
            /* Card Styling for Metrics */
            div[data-testid="stMetric"] {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 10px;
                text-align: center;
            }
        </style>
        
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="theme-color" content="#00CC66">
    """, unsafe_allow_html=True)

# ==========================================
# 2. INIT CLIENTS & SECRETS
# ==========================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    HF_TOKEN = st.secrets.get("HF_TOKEN", "")
except FileNotFoundError:
    st.error("🚨 Critical Error: Secrets file not found.")
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
# 3. SESSION & NAVIGATION
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
        return f"⚠️ Logic Error: {str(e)}"

def get_best_gemini_model():
    """Finds available Gemini model to prevent 404s"""
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: return m.name
                if 'pro' in m.name: return m.name
    except: pass
    return "gemini-1.5-flash"

def analyze_image_robust(image_bytes):
    """
    TRIPLE LAYER SAFETY SYSTEM:
    1. Gemini (Auto-Detected Model)
    2. Hugging Face (BLIP)
    3. Manual Mode Signal
    """
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

# --- GAMIFICATION ---
def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    st.session_state.xp += amount
    today = str(datetime.date.today())
    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp}).eq("user_id", st.session_state.user_id).execute()
        supabase.table("study_logs").insert({
            "user_id": st.session_state.user_id, "minutes": amount, "activity_type": activity_name, "date": today
        }).execute()
        st.toast(f"🌱 +{amount} Pts!", icon="🎉")
        
        # Streak Logic
        if st.session_state.last_action_date != today:
            new_streak = st.session_state.streak + 1
            st.session_state.streak = new_streak
            st.session_state.last_action_date = today
            supabase.table("user_stats").update({"streak": new_streak, "last_study_date": today}).eq("user_id", st.session_state.user_id).execute()
    except: pass

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

# ==========================================
# 5. FEATURE RENDERERS
# ==========================================

def render_home():
    st.write("") 
    st.title(f"👋 Hi, Eco-Warrior")
    
    # 1. Dashboard Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Points", st.session_state.xp)
    c2.metric("Streak", f"{st.session_state.streak}🔥")
    c3.metric("Level", st.session_state.xp // 100)
    
    st.divider()
    
    # 2. Action Grid
    st.subheader("🚀 Start Action")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("♻️ Recycling Bot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
    with c2:
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
        if st.button("🌳 My Forest"): navigate_to("🌳 My Forest")

    st.divider()
    
    # 3. Challenges
    st.subheader("🎯 Today's Goals")
    today = str(datetime.date.today())
    if st.session_state.last_challenge_date != today:
        possible = ["Refill Water Bottle", "Recycle 2 Items", "Meat-Free Meal", "Unplug Devices"]
        st.session_state.daily_challenges = random.sample(possible, 3)
        st.session_state.last_challenge_date = today

    for i, task in enumerate(st.session_state.daily_challenges):
        c_a, c_b = st.columns([4, 1])
        c_a.write(f"✅ **{task}**")
        if c_b.button("Done", key=f"d_{i}"):
            add_xp(20, "Daily Challenge")
            st.balloons()

def render_visual_sorter():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Waste Sorter")
    st.info("Take a photo. AI will tell you how to recycle it.")
    
    t1, t2 = st.tabs(["Camera", "Upload"])
    img_data = None
    with t1:
        cam = st.camera_input("Snap Photo")
        if cam: img_data = cam.getvalue()
    with t2:
        up = st.file_uploader("Upload", type=['jpg','png','jpeg'])
        if up: img_data = up.getvalue(); st.image(img_data, width=200)

    if img_data:
        with st.spinner("Analyzing..."):
            res = analyze_image_robust(img_data)
            if res == "MANUAL_FALLBACK":
                st.warning("⚠️ AI busy. Describe item:")
                man = st.text_input("Item Name")
                if man and st.button("Check"):
                    st.markdown(ask_groq(f"Recycle instructions for {man}"))
                    add_xp(15, "Manual Scan")
            else:
                st.success("✅ Identified!")
                st.markdown(res)
                add_xp(15, "Visual Scan")

def render_map():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Interactive Eco-Map")
    st.info("👆 Click map to Pin Location")

    # Map Logic
    m = folium.Map(location=[20.59, 78.96], zoom_start=4)
    pts = supabase.table("map_points").select("*").execute().data
    for p in pts:
        folium.Marker([p['latitude'], p['longitude']], popup=p['name'], icon=folium.Icon(color="green")).add_to(m)

    map_data = st_folium(m, height=350, width="100%")
    
    # Click Handler
    lat, lon = 20.59, 78.96
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.caption(f"📍 Selected: {lat:.4f}, {lon:.4f}")

    with st.form("pin"):
        n = st.text_input("Location Name")
        t = st.selectbox("Type", ["Recycle Bin", "E-Waste", "Donation"])
        if st.form_submit_button("📍 Pin Spot"):
            supabase.table("map_points").insert({"user_id": st.session_state.user_id, "name": n, "latitude": lat, "longitude": lon, "type": t}).execute()
            st.success("Pinned!")
            time.sleep(1); st.rerun()

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

def render_virtual_forest():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🌳 My Virtual Forest")
    
    trees = st.session_state.xp // 100
    st.metric("Trees Planted", trees)
    
    if trees == 0: st.markdown("# 🌱")
    elif trees < 5: st.markdown(f"# {'🌲 ' * trees}")
    else: st.markdown(f"# {'🌳 ' * trees}")
    
    st.caption("100 Points = 1 Tree. Keep going!")

def render_voice_mode():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    audio = st.audio_input("Ask me anything")
    if audio:
        txt = transcribe_audio(audio)
        st.write(f"You: {txt}")
        st.markdown(ask_groq(txt))
        add_xp(10, "Voice")

def render_recycle_assistant():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Chatbot")
    with st.expander("Upload Guidelines PDF"):
        f = st.file_uploader("PDF", type=['pdf'])
        if f: st.session_state.waste_guidelines_text = extract_text_from_pdf(f)
    q = st.chat_input("Ask...")
    if q:
        ctx = st.session_state.waste_guidelines_text or ""
        st.markdown(ask_groq(q + f"\nContext: {ctx[:5000]}"))
        add_xp(5, "Chat")

def render_mistake_explainer():
    st.write(""); 
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("❌ Mistake Explainer")
    m = st.text_input("I threw...")
    b = st.selectbox("Into...", ["Recycle", "Compost", "Trash"])
    if st.button("Check"):
        st.markdown(ask_groq(f"I put {m} into {b}. Is that bad?"))
        add_xp(10, "Learning")

# ==========================================
# 6. AUTH & MAIN LOOP
# ==========================================
def main():
    apply_mobile_styles()
    
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        t1, t2 = st.tabs(["Login", "Sign Up"])
        with t1:
            e = st.text_input("Email", key="l_e")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Login"): 
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.session_state.user = res.user; st.session_state.user_id = res.user.id
                    sync_user_stats(res.user.id); st.rerun()
                except Exception as err: st.error(str(err))
        with t2:
            e2 = st.text_input("Email", key="s_e")
            p2 = st.text_input("Password", type="password", key="s_p")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: 
                        supabase.table("user_stats").insert({"user_id": res.user.id}).execute()
                        st.success("Account created! Please Login.")
                except Exception as err: st.error(str(err))
        return

    # Sidebar Navigation
    with st.sidebar:
        st.title("EcoWise")
        st.caption(f"User: {st.session_state.user.email}")
        st.divider()
        if st.button("🏠 Home"): navigate_to("🏠 Home")
        if st.button("📸 Visual Sorter"): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode"): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Chatbot"): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map"): navigate_to("🗺️ Eco-Map")
        if st.button("🌊 Plastic Calc"): navigate_to("🌊 Plastic Calculator")
        if st.button("🎨 Upcycling"): navigate_to("🎨 Upcycling Station")
        if st.button("🥗 Eco-Menu"): navigate_to("🥗 Eco-Menu Planner")
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

if __name__ == "__main__":
    main()

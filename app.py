import streamlit as st
import datetime
import time
import os
import json
import re
from supabase import create_client
from groq import Groq
from PyPDF2 import PdfReader

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
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <style>
            footer {visibility: hidden;}
            div.block-container {
                padding-top: max(3.5rem, env(safe-area-inset-top));
                padding-bottom: 5rem;
            }
            /* Make buttons look better on mobile */
            div.stButton > button {
                width: 100%;
                border-radius: 8px;
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
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_groq():
    return Groq(api_key=GROQ_API_KEY)

supabase = init_supabase()
groq_client = init_groq()

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
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

init_session_state()

# Navigation Helper (Used for Sidebar)
def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 3. BACKEND HELPERS
# ==========================================
def ask_ai(prompt, system_role="You are a helpful Sustainability Expert."):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return None

# --- AUTH & DB SYNC ---
def login_user(email, password):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = response.user
        st.session_state.user_id = response.user.id
        sync_user_stats(response.user.id)
        st.success("Welcome back!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

def signup_user(email, password):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            supabase.table("user_stats").insert({
                "user_id": response.user.id,
                "xp": 0,
                "streak": 0
            }).execute()
            st.success("Account created! Please log in.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

def logout_user():
    supabase.auth.sign_out()
    st.session_state.clear()
    st.rerun()

def sync_user_stats(user_id):
    try:
        data = supabase.table("user_stats").select("*").eq("user_id", user_id).execute()
        if data.data:
            stats = data.data[0]
            st.session_state.xp = stats.get('xp', 0)
            
            last_date_str = stats.get('last_study_date')
            db_streak = stats.get('streak', 0)
            
            if last_date_str:
                last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                today = datetime.date.today()
                gap = (today - last_date).days
                if gap > 1: st.session_state.streak = 0
                else: st.session_state.streak = db_streak
                st.session_state.last_action_date = last_date_str
            else:
                st.session_state.streak = 0
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
            "user_id": st.session_state.user_id,
            "minutes": amount, 
            "activity_type": activity_name,
            "date": today
        }).execute()
        st.toast(f"🌱 +{amount} Green Points!", icon="🌍")
        
        if st.session_state.last_action_date != today:
            new_streak = st.session_state.streak + 1
            st.session_state.streak = new_streak
            st.session_state.last_action_date = today
            supabase.table("user_stats").update({"streak": new_streak, "last_study_date": today}).eq("user_id", st.session_state.user_id).execute()
    except Exception as e:
        st.error(f"Sync Error: {e}")

# ==========================================
# 4. FEATURE RENDERERS
# ==========================================

def render_home():
    st.write("") 
    st.title("🌍 EcoWise Dashboard")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Points", st.session_state.xp)
    c2.metric("🔥 Streak", f"{st.session_state.streak} Days")
    level = "Eco-Warrior" if st.session_state.xp > 500 else "Rookie"
    c3.metric("🏆 Rank", level)
    
    st.divider()
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("🕵️ Greenwash Detector", use_container_width=True): navigate_to("🕵️ Greenwash Detector")
    with col2:
        if st.button("👣 Carbon Tracker", use_container_width=True): navigate_to("👣 Carbon Tracker")
        if st.button("🎮 Eco-Challenges", use_container_width=True): navigate_to("🎮 Eco-Challenges")

def render_recycle_assistant():
    st.write("")
    # FIX: Direct Logic - Single Click Works
    if st.button("⬅️ Back to Home"):
        navigate_to("🏠 Home")
    
    st.header("♻️ Smart Recycle Assistant")
    st.info("Upload your local city/campus waste guidelines (PDF) to get accurate answers.")
    
    with st.expander("📂 Upload Municipal Guidelines (PDF)", expanded=False):
        uploaded_file = st.file_uploader("Upload Waste Guide PDF", type=['pdf'])
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.session_state.waste_guidelines_text = text
                st.success("✅ Guidelines Loaded!")
    
    user_query = st.chat_input("E.g., Can I recycle pizza boxes?")
    if user_query:
        with st.spinner("Consulting Guidelines..."):
            system_role = "You are a waste management expert. Use the provided guidelines if available."
            if st.session_state.waste_guidelines_text:
                system_role += f"\n\nOFFICIAL GUIDELINES:\n{st.session_state.waste_guidelines_text[:15000]}"
            
            response = ask_ai(user_query, system_role)
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(response)
            add_xp(5, "Waste Query")

def render_greenwash_detector():
    st.write("")
    if st.button("⬅️ Back to Home"):
        navigate_to("🏠 Home")
    
    st.header("🕵️ Greenwash Detector")
    st.write("Paste a product description to check for eco-authenticity.")
    
    product_text = st.text_area("Product Claim")
    if st.button("Analyze Claim"):
        if product_text:
            with st.spinner("Auditing..."):
                prompt = (f"Analyze this product claim for 'Greenwashing'. Claim: '{product_text}'\n"
                          "1. Is it vague? 2. Proof? 3. Verdict: Greenwashed or Genuine?")
                analysis = ask_ai(prompt)
                st.markdown(analysis)
                add_xp(10, "Greenwash Check")
        else:
            st.warning("Enter text first.")

def render_carbon_tracker():
    st.write("")
    if st.button("⬅️ Back to Home"):
        navigate_to("🏠 Home")
    
    st.header("👣 Daily Carbon Tracker")
    transport = st.selectbox("Transport", ["Walk/Cycle", "Bus/Train", "Car (Petrol)", "Car (EV)"])
    meal = st.selectbox("Meal", ["Plant-based", "Vegetarian", "Meat-heavy"])
    energy = st.checkbox("Saved Energy?")
    
    if st.button("Calculate Impact"):
        with st.spinner("Calculating..."):
            score = 0
            if transport == "Walk/Cycle": score += 20
            elif transport == "Car (Petrol)": score -= 10
            if meal == "Plant-based": score += 20
            elif meal == "Meat-heavy": score -= 10
            if energy: score += 10
            
            prompt = f"User did: {transport}, {meal}, Energy Saved: {energy}. Give 1 short eco-tip."
            ai_tip = ask_ai(prompt)
            
            st.success(f"Score: {score}/50")
            st.info(f"💡 Tip: {ai_tip}")
            if score > 0: add_xp(score, "Daily Carbon Log")

def render_challenges():
    st.write("")
    if st.button("⬅️ Back to Home"):
        navigate_to("🏠 Home")
    
    st.header("🎮 Eco-Challenges")
    challenges = [
        {"task": "Reusable Bottle", "xp": 20},
        {"task": "Refuse Plastic Bag", "xp": 15},
        {"task": "Segregate Waste", "xp": 25},
        {"task": "Save Water", "xp": 10}
    ]
    for c in challenges:
        c1, c2 = st.columns([3,1])
        c1.write(f"**{c['task']}**")
        if c2.button(f"Claim +{c['xp']}", key=c['task']):
            add_xp(c['xp'], c['task'])
            st.balloons()

# ==========================================
# 5. MAIN NAV
# ==========================================
def main():
    make_pwa_ready()
    
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        t1, t2 = st.tabs(["Login", "Sign Up"])
        with t1:
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Login"): login_user(e, p)
        with t2:
            e2 = st.text_input("Email (Sign Up)")
            p2 = st.text_input("Password (Sign Up)", type="password")
            if st.button("Sign Up"): signup_user(e2, p2)
        return

    # Sidebar
    with st.sidebar:
        st.title("EcoWise AI")
        st.caption("Powered by Llama 3 & IBM Granite Ready")
        st.write(f"👤 {st.session_state.user.email}")
        
        if st.button("🏠 Home", use_container_width=True): navigate_to("🏠 Home")
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("🕵️ Greenwash Detector", use_container_width=True): navigate_to("🕵️ Greenwash Detector")
        if st.button("👣 Carbon Tracker", use_container_width=True): navigate_to("👣 Carbon Tracker")
        if st.button("🎮 Eco-Challenges", use_container_width=True): navigate_to("🎮 Eco-Challenges")
        
        st.divider()
        if st.button("🚪 Logout"): logout_user()

    # Routing
    if st.session_state.feature == "🏠 Home": render_home()
    elif st.session_state.feature == "♻️ Recycle Assistant": render_recycle_assistant()
    elif st.session_state.feature == "🕵️ Greenwash Detector": render_greenwash_detector()
    elif st.session_state.feature == "👣 Carbon Tracker": render_carbon_tracker()
    elif st.session_state.feature == "🎮 Eco-Challenges": render_challenges()

if __name__ == "__main__":
    main()

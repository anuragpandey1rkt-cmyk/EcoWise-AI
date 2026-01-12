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
            .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
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
        "chat_history": [],
        "waste_guidelines_text": "" # Stores the PDF text for RAG
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def go_to(page):
    st.session_state.feature = page

# ==========================================
# 3. BACKEND HELPERS
# ==========================================

def ask_ai(prompt, system_role="You are a helpful Sustainability Expert."):
    try:
        # Note: We use Llama 3 here for speed, but this slot is compatible with IBM Granite
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
        st.success("Welcome back, Eco Warrior!")
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
            
            # Streak Logic
            last_date_str = stats.get('last_study_date') # Reusing column name 'last_study_date' for simplicity
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
            # Create row
            supabase.table("user_stats").insert({"user_id": user_id, "xp": 0, "streak": 0}).execute()
    except Exception as e:
        print(f"Sync Error: {e}")

def add_xp(amount, activity_name):
    if not st.session_state.user_id: return
    
    st.session_state.xp += amount
    today = str(datetime.date.today())
    
    # Update DB
    try:
        supabase.table("user_stats").update({"xp": st.session_state.xp}).eq("user_id", st.session_state.user_id).execute()
        
        # Log Activity (Reusing study_logs table but storing eco-actions)
        supabase.table("study_logs").insert({
            "user_id": st.session_state.user_id,
            "minutes": 1, # Just a placeholder
            "activity_type": activity_name,
            "date": today
        }).execute()
        
        st.toast(f"🌱 +{amount} Green Points! ({activity_name})", icon="🌍")
        
        # Streak Update
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
    st.title("🌍 EcoWise Dashboard")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Green Points", st.session_state.xp)
    c2.metric("🔥 Eco Streak", f"{st.session_state.streak} Days")
    
    level = "Eco-Warrior" if st.session_state.xp > 500 else "Sustainability Rookie"
    c3.metric("🏆 Status", level)
    
    st.divider()
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("♻️ Recycle Assistant", use_container_width=True, on_click=go_to, args=("♻️ Recycle Assistant",))
        st.button("🕵️ Greenwash Detector", use_container_width=True, on_click=go_to, args=("🕵️ Greenwash Detector",))
    with col2:
        st.button("👣 Carbon Tracker", use_container_width=True, on_click=go_to, args=("👣 Carbon Tracker",))
        st.button("🎮 Eco-Challenges", use_container_width=True, on_click=go_to, args=("🎮 Eco-Challenges",))

def render_recycle_assistant():
    st.header("♻️ Smart Recycle Assistant")
    st.info("Upload your local city/campus waste guidelines (PDF) to get accurate answers.")
    
    # PDF Upload for RAG
    with st.expander("📂 Upload Municipal Guidelines (PDF)", expanded=False):
        uploaded_file = st.file_uploader("Upload Waste Guide PDF", type=['pdf'])
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.session_state.waste_guidelines_text = text
                st.success("✅ Guidelines Loaded! AI is now locally aware.")
    
    # Chat Interface
    user_query = st.chat_input("E.g., Can I recycle pizza boxes?")
    
    if user_query:
        with st.spinner("Analyzing..."):
            system_role = "You are a waste management expert. Use the provided guidelines if available."
            if st.session_state.waste_guidelines_text:
                system_role += f"\n\nOFFICIAL GUIDELINES:\n{st.session_state.waste_guidelines_text[:15000]}"
            
            response = ask_ai(user_query, system_role)
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(response)
            
            # XP for asking
            add_xp(5, "Waste Query")

def render_greenwash_detector():
    st.header("🕵️ Greenwash Detector")
    st.write("Paste a product description or marketing claim. AI will analyze if it's truly eco-friendly or just marketing hype.")
    
    product_text = st.text_area("Product Claim (e.g., '100% Natural Organic Bottle')")
    
    if st.button("Analyze Claim"):
        if product_text:
            with st.spinner("Auditing claim..."):
                prompt = (
                    f"Analyze this product claim for 'Greenwashing'. \n"
                    f"Claim: '{product_text}'\n"
                    f"1. Is it vague? \n"
                    f"2. Are there proof/certifications? \n"
                    f"3. Verdict: Greenwashed or Genuine? \n"
                    f"Provide a strict, fact-based analysis."
                )
                analysis = ask_ai(prompt)
                st.markdown("### 🔍 AI Audit Report")
                st.markdown(analysis)
                add_xp(10, "Greenwash Check")
        else:
            st.warning("Please enter text first.")

def render_carbon_tracker():
    st.header("👣 Daily Carbon Quick-Check")
    st.write("Log your daily actions to estimate impact.")
    
    transport = st.selectbox("Transport used today", ["Walk/Cycle", "Public Bus/Train", "Car (Petrol)", "Car (EV)"])
    meal = st.selectbox("Main Meal type", ["Plant-based", "Vegetarian", "Meat-heavy"])
    energy = st.checkbox("Did you turn off lights/AC when leaving?")
    
    if st.button("Calculate Impact"):
        # Simple heuristic logic
        score = 0
        feedback = ""
        
        if transport == "Walk/Cycle": score += 20; feedback += "✅ Great low-carbon transport! "
        elif transport == "Car (Petrol)": score -= 10; feedback += "⚠️ Car travel has high emissions. "
        
        if meal == "Plant-based": score += 20; feedback += "✅ Low water/carbon footprint meal. "
        elif meal == "Meat-heavy": score -= 10; feedback += "⚠️ Meat production has high impact. "
        
        if energy: score += 10; feedback += "✅ Saved energy. "
        
        st.success(f"Daily Score: {score}/50")
        st.write(f"**Insight:** {feedback}")
        
        if score > 0:
            add_xp(score, "Daily Carbon Log")

def render_challenges():
    st.header("🎮 Eco-Challenges")
    st.info("Complete these actions today to earn Green Points!")
    
    challenges = [
        {"task": "Use a reusable water bottle", "xp": 20},
        {"task": "Refuse a plastic bag", "xp": 15},
        {"task": "Segregate wet & dry waste correctly", "xp": 25},
        {"task": "Turn off tap while brushing", "xp": 10}
    ]
    
    for c in challenges:
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{c['task']}**")
        if col2.button(f"Claim +{c['xp']}", key=c['task']):
            add_xp(c['xp'], c['task'])
            st.balloons()

# ==========================================
# 5. MAIN NAV
# ==========================================
def main():
    make_pwa_ready()
    
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Login"): login_user(e, p)
        with tab2:
            e2 = st.text_input("Email (Sign Up)")
            p2 = st.text_input("Password (Sign Up)", type="password")
            if st.button("Sign Up"): signup_user(e2, p2)
        return

    # Sidebar
    with st.sidebar:
        st.title("EcoWise AI")
        st.caption("Powered by Llama 3 & IBM Granite Ready") # <--- CREDITS TO IBM
        st.write(f"👤 {st.session_state.user.email}")
        
        opts = ["🏠 Home", "♻️ Recycle Assistant", "🕵️ Greenwash Detector", "👣 Carbon Tracker", "🎮 Eco-Challenges"]
        for o in opts:
            if st.button(o, use_container_width=True):
                go_to(o)
                st.rerun()
                
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

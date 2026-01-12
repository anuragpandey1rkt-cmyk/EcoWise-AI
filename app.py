import streamlit as st
import datetime
import time
import base64
import pandas as pd
from supabase import create_client
from groq import Groq
from PyPDF2 import PdfReader
import folium
from streamlit_folium import st_folium

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

def navigate_to(page):
    st.session_state.feature = page
    st.rerun()

# ==========================================
# 3. AI & HELPER FUNCTIONS
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

def analyze_image(image_bytes):
    """Uses Llama 3.2 Vision to analyze trash/items"""
    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        completion = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview", # <--- UPDATED TO NEW MODEL
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify this object. Is it recyclable, compostable, or trash? Be brief and give strict disposal instructions."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Vision Error: {str(e)}"

def transcribe_audio(audio_bytes):
    """Uses Whisper to transcribe voice"""
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
            st.session_state.last_action_date = stats.get('last_study_date')
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
        
        # Streak Logic
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
    c3.metric("🏆 Rank", "Eco-Warrior" if st.session_state.xp > 500 else "Rookie")
    
    st.divider()
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📸 Visual Sorter", use_container_width=True): navigate_to("📸 Visual Sorter")
        if st.button("🎙️ Voice Mode", use_container_width=True): navigate_to("🎙️ Voice Mode")
        if st.button("♻️ Recycle Assistant", use_container_width=True): navigate_to("♻️ Recycle Assistant")
        if st.button("🗺️ Eco-Map", use_container_width=True): navigate_to("🗺️ Eco-Map")
    with col2:
        if st.button("🛒 Campus Swap", use_container_width=True): navigate_to("🛒 Campus Swap")
        if st.button("📊 Green Analytics", use_container_width=True): navigate_to("📊 Green Analytics")
        if st.button("🕵️ Greenwash Detector", use_container_width=True): navigate_to("🕵️ Greenwash Detector")
        if st.button("👣 Carbon Tracker", use_container_width=True): navigate_to("👣 Carbon Tracker")

def render_visual_sorter():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📸 AI Visual Waste Sorter")
    st.info("Take a photo of trash. The AI will tell you which bin to use.")
    
    img_file = st.camera_input("Take a picture")
    
    if img_file:
        bytes_data = img_file.getvalue()
        with st.spinner("Analyzing image..."):
            result = analyze_image(bytes_data)
            st.success("Analysis Complete!")
            st.markdown(result)
            add_xp(15, "Visual Scan")

def render_voice_mode():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🎙️ Voice Assistant")
    st.write("Ask any eco-question (English/Hindi) using your voice.")
    
    audio_value = st.audio_input("Record your question")
    
    if audio_value:
        with st.spinner("Listening & Thinking..."):
            # 1. Transcribe
            text_query = transcribe_audio(audio_value)
            st.write(f"**You said:** {text_query}")
            
            # 2. Get Answer
            response = ask_ai(text_query)
            st.markdown(f"**AI:** {response}")
            add_xp(10, "Voice Query")

def render_map():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🗺️ Community Eco-Map")
    st.write("Find or Add local recycling spots.")
    
    # 1. Add Point Form
    with st.expander("📍 Add a New Spot"):
        with st.form("map_form"):
            name = st.text_input("Location Name (e.g., Library Battery Bin)")
            lat = st.number_input("Latitude", value=28.6139, format="%.4f")
            lon = st.number_input("Longitude", value=77.2090, format="%.4f")
            ptype = st.selectbox("Type", ["Recycling", "Donation", "Hazardous Waste"])
            if st.form_submit_button("Add Pin"):
                supabase.table("map_points").insert({
                    "user_id": st.session_state.user_id,
                    "name": name, "latitude": lat, "longitude": lon, "type": ptype
                }).execute()
                st.success("Pin Added!")
                add_xp(20, "Added Map Pin")
                st.rerun()

    # 2. Display Map
    points = supabase.table("map_points").select("*").execute().data
    
    # Center map on first point or default (India)
    start_loc = [points[0]['latitude'], points[0]['longitude']] if points else [20.5937, 78.9629]
    m = folium.Map(location=start_loc, zoom_start=10)
    
    for p in points:
        icon_color = "green" if p['type'] == "Recycling" else "blue"
        folium.Marker(
            [p['latitude'], p['longitude']], 
            popup=p['name'], 
            tooltip=p['type'],
            icon=folium.Icon(color=icon_color, icon="leaf")
        ).add_to(m)
    
    st_folium(m, width="100%", height=400)

def render_marketplace():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🛒 Campus Swap (Circular Economy)")
    st.write("Don't throw it away. Give it away!")
    
    tab1, tab2 = st.tabs(["Browse Items", "Sell/Donate Item"])
    
    with tab1:
        items = supabase.table("marketplace_items").select("*").order("created_at", desc=True).execute().data
        if items:
            for i in items:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.subheader(i['item_name'])
                    c1.write(i['description'])
                    c1.caption(f"Contact: {i['contact_info']}")
                    c2.metric("Price", i['price'])
        else:
            st.info("No items yet. Be the first to list something!")

    with tab2:
        with st.form("sell_form"):
            name = st.text_input("Item Name")
            desc = st.text_area("Description")
            price = st.text_input("Price (or 'Free')")
            contact = st.text_input("Your Contact (Phone/Email)")
            if st.form_submit_button("List Item"):
                supabase.table("marketplace_items").insert({
                    "user_id": st.session_state.user_id,
                    "item_name": name, "description": desc, "price": price, "contact_info": contact
                }).execute()
                st.success("Item Listed!")
                add_xp(30, "Listed Item")
                st.rerun()

def render_analytics():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("📊 Green Analytics")
    
    # Fetch Data
    logs = supabase.table("study_logs").select("*").execute().data
    
    if logs:
        df = pd.DataFrame(logs)
        
        # 1. Total Impact
        total_points = df['minutes'].sum() # Reusing minutes column for points
        st.metric("Total Community Green Points", total_points)
        
        # 2. Activity Distribution
        st.subheader("Most Popular Eco-Activities")
        chart_data = df['activity_type'].value_counts()
        st.bar_chart(chart_data)
        
        # 3. Timeline
        st.subheader("Community Activity Over Time")
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        daily_activity = df.groupby('date')['minutes'].sum()
        st.line_chart(daily_activity)
    else:
        st.info("Not enough data for analytics yet.")

def render_recycle_assistant():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("♻️ Smart Recycle Assistant")
    
    with st.expander("📂 Upload Municipal Guidelines (PDF)", expanded=False):
        uploaded_file = st.file_uploader("Upload Waste Guide PDF", type=['pdf'])
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.session_state.waste_guidelines_text = text
                st.success("✅ Guidelines Loaded!")
    
    user_query = st.chat_input("E.g., Can I recycle pizza boxes?")
    if user_query:
        with st.spinner("Thinking..."):
            system_role = "You are a waste management expert. Use the provided guidelines if available."
            if st.session_state.waste_guidelines_text:
                system_role += f"\n\nOFFICIAL GUIDELINES:\n{st.session_state.waste_guidelines_text[:15000]}"
            res = ask_ai(user_query, system_role)
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(res)
            add_xp(5, "Waste Query")

def render_greenwash_detector():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("🕵️ Greenwash Detector")
    txt = st.text_area("Product Claim")
    if st.button("Analyze") and txt:
        with st.spinner("Auditing..."):
            st.write(ask_ai(f"Analyze greenwashing: {txt}"))
            add_xp(10, "Greenwash Check")

def render_carbon_tracker():
    st.write("")
    if st.button("⬅️ Back"): navigate_to("🏠 Home")
    st.header("👣 Carbon Tracker")
    
    t = st.selectbox("Transport", ["Walk/Cycle", "Car", "Bus"])
    if st.button("Log Sustainable Action"):
        add_xp(20, f"Transport: {t}")
        st.success("Logged!")

# ==========================================
# 5. MAIN NAV
# ==========================================
def main():
    make_pwa_ready()
    
    # Login Logic
    if not st.session_state.user:
        st.title("🌱 EcoWise Login")
        t1, t2 = st.tabs(["Login", "Sign Up"])
        with t1:
            e = st.text_input("Email"); p = st.text_input("Password", type="password")
            if st.button("Login"): 
                try:
                    res = supabase.auth.sign_in_with_password({"email": e, "password": p})
                    st.session_state.user = res.user; st.session_state.user_id = res.user.id
                    sync_user_stats(res.user.id); st.rerun()
                except Exception as err: st.error(str(err))
        with t2:
            e2 = st.text_input("Email (Sign Up)"); p2 = st.text_input("Password (Sign Up)", type="password")
            if st.button("Sign Up"):
                try:
                    res = supabase.auth.sign_up({"email": e2, "password": p2})
                    if res.user: 
                        supabase.table("user_stats").insert({"user_id": res.user.id}).execute()
                        st.success("Created! Login now.")
                except Exception as err: st.error(str(err))
        return

    # Routing
    f = st.session_state.feature
    if f == "🏠 Home": render_home()
    elif f == "📸 Visual Sorter": render_visual_sorter()
    elif f == "🎙️ Voice Mode": render_voice_mode()
    elif f == "🗺️ Eco-Map": render_map()
    elif f == "🛒 Campus Swap": render_marketplace()
    elif f == "📊 Green Analytics": render_analytics()
    elif f == "♻️ Recycle Assistant": render_recycle_assistant()
    elif f == "🕵️ Greenwash Detector": render_greenwash_detector()
    elif f == "👣 Carbon Tracker": render_carbon_tracker()

if __name__ == "__main__":
    main()

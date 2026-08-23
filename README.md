## 🌱 EcoWise AI – Hyper-Local Sustainability Assistant

🔗 **Live App:** https://ecowise-ai-2026.streamlit.app/  
📦 **Source Code:** https://github.com/anuragpandey1rkt-cmyk/EcoWise-AI

🌱 EcoWise AI – Hyper-Local Sustainability Assistant

EcoWise AI is a full-stack, AI-powered sustainability platform built with Streamlit that helps users make smarter, eco-friendly decisions through AI vision, voice, maps, gamification, and analytics.

The app combines Generative AI, Computer Vision, Voice AI, Maps, and Gamification to promote responsible consumption and environmental awareness.

🚀 Key Features
🏠 Home Dashboard

XP, streak, rank tracking

Quick navigation to all eco-tools

Gamified user experience

📸 AI Waste Visual Sorter

Take a photo or upload an image

AI identifies the item and explains how to recycle it

Multi-layer AI fallback system:

Google Gemini (Primary)

Hugging Face (Backup)

Manual input (Final fallback)

🎙️ Voice Mode

Ask sustainability questions using your voice

Uses Whisper (Groq) for speech-to-text

AI responds instantly

♻️ Recycling Assistant (RAG-like)

Upload official recycling PDFs

Ask questions based on uploaded rules

AI combines document context + user query

🗺️ Interactive Eco-Map

Click anywhere on the map to pin eco-locations

Add:

Recycling bins

E-waste centers

Donation points

Compost locations

Uses Folium + Supabase

🌊 Plastic Footprint Calculator

Calculates annual plastic waste

Categorizes impact:

Low

Medium

High

Generates AI-based reduction strategies

🎨 Upcycling Station

Enter any waste item

Get creative DIY reuse ideas

Encourages circular economy thinking

🥗 Low-Carbon Menu Planner

AI-generated eco-friendly meals

Supports multiple cuisines

Explains environmental benefits

🌳 My Virtual Forest (Gamification)

Earn 1 tree for every 100 XP

Visual forest grows with your eco-actions

Shows recent activity logs

👣 Carbon Footprint Saver

Track CO₂ saved by choosing eco-friendly transport

Calculates carbon emissions prevented

XP rewards based on impact

❌ Mistake Explainer

Learn the environmental impact of wrong disposal

Encourages behavior correction instead of punishment

🏆 Leaderboard

Global ranking based on XP

Encourages healthy competition

🧠 AI Architecture (Triple-Layer Design)
Layer	Technology
Vision AI	Google Gemini / Hugging Face
LLM Reasoning	Groq (LLaMA 3.3 70B)
Voice AI	Whisper (Groq)
Fallbacks	Manual + Rule-based
🛠️ Tech Stack
Frontend

Streamlit

Custom CSS for PWA-like UI

Backend & Database

Supabase (Auth + PostgreSQL)

Secure session handling

AI & ML

Groq API (LLMs + Whisper)

Google Gemini Vision

Hugging Face Inference API

Maps & Visualization

Folium

Plotly

📦 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ecowise-ai.git
cd ecowise-ai

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Secrets

Create a file:

.streamlit/secrets.toml


Add:

SUPABASE_URL="your_supabase_url"
SUPABASE_ANON_KEY="your_supabase_anon_key"
GROQ_API_KEY="your_groq_api_key"
GEMINI_API_KEY="your_gemini_api_key"
HF_TOKEN="your_huggingface_token"

4️⃣ Run the App
streamlit run app.py

🗄️ Database Tables (Supabase)

Required tables:

user_stats

study_logs

map_points

Example fields:

user_id, xp, streak, last_study_date
latitude, longitude, name, type

🎯 Use Cases

Sustainability education

Smart waste management

College / internship projects

Hackathons & SDG initiatives

AI portfolio showcase

🌍 SDG Alignment

SDG 12 – Responsible Consumption & Production

SDG 11 – Sustainable Cities & Communities

SDG 13 – Climate Action

👨‍💻 Author

Anurag Pandey
AI & Full-Stack Developer
🌱 Passionate about sustainability, AI, and real-world impact

⭐ Future Enhancements

Mobile PWA deployment

Community challenges

NGO & municipality integrations

Real-time IoT waste data

Rewards & badges system


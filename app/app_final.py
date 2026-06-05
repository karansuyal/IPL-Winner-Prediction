import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="IPL Predictor", page_icon="🏏")
st.title("🏏 IPL Match Winner Predictor")
st.markdown("### For Celebal Technologies Campus Placement")

# Direct paths (absolute path)
BASE = r'C:\Users\Suyal\OneDrive\Desktop\ML Project\models'

@st.cache_resource
def load_models():
    model = joblib.load(f'{BASE}/ipl_model.pkl')
    le_team_a = joblib.load(f'{BASE}/le_team_a.pkl')
    le_team_b = joblib.load(f'{BASE}/le_team_b.pkl')
    le_venue = joblib.load(f'{BASE}/le_venue.pkl')
    le_toss_winner = joblib.load(f'{BASE}/le_toss_winner.pkl')
    le_toss_decision = joblib.load(f'{BASE}/le_toss_decision.pkl')
    le_winner = joblib.load(f'{BASE}/le_winner.pkl')
    return model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner

try:
    model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner = load_models()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Please run the training notebook first to save the model.")
    st.stop()

all_teams = list(le_team_a.classes_)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", all_teams)
with col2:
    team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1])

venue = st.selectbox("Venue", le_venue.classes_)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

if st.button("Predict Winner", type="primary"):
    t1 = le_team_a.transform([team1])[0]
    t2 = le_team_b.transform([team2])[0]
    v = le_venue.transform([venue])[0]
    tw = le_toss_winner.transform([toss_winner])[0]
    td = le_toss_decision.transform([toss_decision])[0]
    
    features = np.array([[t1, t2, v, tw, td]])
    pred = model.predict(features)[0]
    winner = le_winner.inverse_transform([pred])[0]
    
    st.markdown("---")
    if winner == team1:
        st.success(f"### 🏆 Winner: {team1}")
    else:
        st.success(f"### 🏆 Winner: {team2}")

st.markdown("---")
st.caption("IPL 2008-2024 Data | Random Forest Model")
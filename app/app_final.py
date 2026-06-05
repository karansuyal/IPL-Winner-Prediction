import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="IPL Predictor", page_icon="🏏")

st.title("🏏 IPL Match Winner Predictor")
st.markdown("### For Celebal Technologies Campus Placement")

# Smart path detection - works on local and cloud
@st.cache_resource
def load_model():
    # Try different possible paths
    possible_paths = [
        'models',
        './models', 
        '../models',
        '/mount/src/ipl-winner-prediction/models',
        '/app/models'
    ]
    
    BASE = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'ipl_model.pkl')):
            BASE = path
            break
    
    if BASE is None:
        st.error("❌ Model files not found! Please check deployment.")
        st.info(f"Current directory: {os.getcwd()}")
        st.info(f"Files in current dir: {os.listdir('.')}")
        if os.path.exists('models'):
            st.info(f"Files in models: {os.listdir('models')}")
        return None, None, None, None, None, None, None
    
    try:
        model = joblib.load(f'{BASE}/ipl_model.pkl')
        le_team_a = joblib.load(f'{BASE}/le_team_a.pkl')
        le_team_b = joblib.load(f'{BASE}/le_team_b.pkl')
        le_venue = joblib.load(f'{BASE}/le_venue.pkl')
        le_toss_winner = joblib.load(f'{BASE}/le_toss_winner.pkl')
        le_toss_decision = joblib.load(f'{BASE}/le_toss_decision.pkl')
        le_winner = joblib.load(f'{BASE}/le_winner.pkl')
        return model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None, None, None

# Load all models
model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner = load_model()

if model is None:
    st.stop()

all_teams = list(le_team_a.classes_)
st.success("✅ Model loaded successfully!")

# Input form
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", all_teams)
    venue = st.selectbox("Venue", le_venue.classes_)

with col2:
    team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1])
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# Predict button
if st.button("Predict Winner", type="primary"):
    # Encode inputs
    t1 = le_team_a.transform([team1])[0]
    t2 = le_team_b.transform([team2])[0]
    v = le_venue.transform([venue])[0]
    tw = le_toss_winner.transform([toss_winner])[0]
    td = le_toss_decision.transform([toss_decision])[0]
    
    # Predict
    features = np.array([[t1, t2, v, tw, td]])
    prediction = model.predict(features)[0]
    winner = le_winner.inverse_transform([prediction])[0]
    
    # Show result
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if winner == team1:
            st.success(f"### 🏆 Winner: {team1}")
        else:
            st.success(f"### 🏆 Winner: {team2}")

# Footer
st.markdown("---")
st.caption("IPL 2008-2024 Data | Random Forest Model | Celebal Technologies")
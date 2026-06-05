import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="IPL Winner Predictor", page_icon="🏏")

st.title("🏏 IPL Match Winner Predictor")
st.markdown("### Predict the winner based on historical IPL data (2008-2024)")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('../models/ipl_model.pkl')
    le_team_a = joblib.load('../models/le_team_a.pkl')
    le_team_b = joblib.load('../models/le_team_b.pkl')
    le_venue = joblib.load('../models/le_venue.pkl')
    le_toss_winner = joblib.load('../models/le_toss_winner.pkl')
    le_toss_decision = joblib.load('../models/le_toss_decision.pkl')
    le_winner = joblib.load('../models/le_winner.pkl')
    return model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner

model, le_team_a, le_team_b, le_venue, le_toss_winner, le_toss_decision, le_winner = load_model()

# Get all teams
all_teams = list(le_team_a.classes_)

st.subheader("📋 Enter Match Details")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", all_teams)
    
with col2:
    team2 = st.selectbox("Team 2", [t for t in all_teams if t != team1])

venue = st.selectbox("🏟️ Venue", le_venue.classes_)
toss_winner = st.selectbox("🎲 Toss Winner", [team1, team2])
toss_decision = st.selectbox("📢 Toss Decision", ["bat", "field"])

if st.button("🔮 Predict Winner", type="primary"):
    # Encode inputs
    team1_enc = le_team_a.transform([team1])[0]
    team2_enc = le_team_b.transform([team2])[0]
    venue_enc = le_venue.transform([venue])[0]
    toss_winner_enc = le_toss_winner.transform([toss_winner])[0]
    toss_decision_enc = le_toss_decision.transform([toss_decision])[0]
    
    # Predict
    features = np.array([[team1_enc, team2_enc, venue_enc, toss_winner_enc, toss_decision_enc]])
    prediction = model.predict(features)[0]
    winner = le_winner.inverse_transform([prediction])[0]
    
    # Show result
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if winner == team1:
            st.success(f"### 🏆 Predicted Winner: **{team1}**")
        else:
            st.success(f"### 🏆 Predicted Winner: **{team2}**")
    
    st.info("⚠️ **Note:** IPL is highly unpredictable. This prediction is based on historical patterns.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model Details:**
    - Data: IPL 2008-2024
    - Matches: 1000+
    - Features: Teams, Venue, Toss
    - Model: Random Forest
    
    **Made for:** Celebal Technologies Campus Placement
    """)

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning | IPL Data 2008-2024")
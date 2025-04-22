import streamlit as st
import pandas as pd
import joblib

# Load your saved model
pipe_xgb = joblib.load('pipe_xgb_model.pkl')

# Title
st.title("ODI Cricket Score Predictor(1st innings)")

# User Inputs
batting_team = st.selectbox("Batting Team", ['Australia','India','Bangladesh','New Zealand','South Africa','England','West Indies','Afghanistan','Pakistan','Sri Lanka'])
bowling_team = st.selectbox("Bowling Team", ['Australia','India','Bangladesh','New Zealand','South Africa','England','West Indies','Afghanistan','Pakistan','Sri Lanka'])
venue = st.selectbox("Venue", [
    'Brisbane Cricket Ground, Woolloongabba',
       'Melbourne Cricket Ground',
       'Western Australia Cricket Association Ground',
       'Sydney Cricket Ground', 'Adelaide Oval', 'Manuka Oval',
       'Hagley Oval', 'Saxton Oval', 'Eden Park', 'Seddon Park',
       'Westpac Stadium', 'Kennington Oval', 'Edgbaston',
       'Sophia Gardens', 'Sir Vivian Richards Stadium, North Sound',
       'Kensington Oval, Bridgetown', 'Shere Bangla National Stadium',
       'Zahur Ahmed Chowdhury Stadium', 'Feroz Shah Kotla',
       'Punjab Cricket Association Stadium, Mohali',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Headingley', 'The Rose Bowl', "Lord's", 'Old Trafford',
       'County Ground', 'Clontarf Cricket Club Ground',
       'Maharashtra Cricket Association Stadium', 'Barabati Stadium',
       'Eden Gardens', 'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium',
       'Harare Sports Club', 'Queens Sports Club', 'Providence Stadium',
       'Rangiri Dambulla International Stadium',
       'Sinhalese Sports Club Ground', "Queen's Park Oval, Port of Spain",
       'Sabina Park, Kingston', 'Pallekele International Cricket Stadium',
       'R.Premadasa Stadium, Khettarama', 'Bay Oval', 'University Oval',
       'MA Chidambaram Stadium, Chepauk',
       'Vidarbha Cricket Association Stadium, Jamtha', 'Trent Bridge',
       'Riverside Ground', 'Wankhede Stadium', 'Green Park',
       'Dubai International Cricket Stadium', 'Kingsmead',
       'SuperSport Park', 'Newlands', 'The Wanderers Stadium',
       "St George's Park", 'Shere Bangla National Stadium, Mirpur',
       'Bellerive Oval', 'Warner Park, Basseterre',
       'Sylhet International Cricket Stadium', 'McLean Park',
       'Rajiv Gandhi International Stadium, Uppal',
       'The Village, Malahide', 'National Stadium',
       'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa',
       'SuperSport Park, Centurion', 'R Premadasa Stadium, Colombo',
       'Kensington Oval, Bridgetown, Barbados', 'Newlands, Cape Town',
       'Narendra Modi Stadium, Ahmedabad', 'Gaddafi Stadium, Lahore',
       'Multan Cricket Stadium', 'National Stadium, Karachi',
       'Rawalpindi Cricket Stadium', 'Eden Gardens, Kolkata',
       'Wankhede Stadium, Mumbai', 'Arnos Vale Ground, Kingstown',
       'Nehru Stadium', 'Gaddafi Stadium', 'New Wanderers Stadium',
       'R Premadasa Stadium', 'Beausejour Stadium, Gros Islet',
       "National Cricket Stadium, St George's", 'Sawai Mansingh Stadium',
       'Sardar Patel Stadium, Motera', 'Kinrara Academy Oval',
       'Civil Service Cricket Club, Stormont', 'M Chinnaswamy Stadium',
       'Willowmoore Park', 'Sharjah Cricket Association Stadium'
])
current_score = st.number_input("Current Score", min_value=0, value=14)
balls_left = st.number_input("Balls Left", min_value=0, max_value=300, value=269)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10, value=8)
crr = st.number_input("Current Run Rate (CRR)", value=2.709677)
last_five = st.number_input("Runs in Last 5 Overs", value=14.0)

# Submit button
if st.button("Predict Final Score"):
    # Build the input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    # Make prediction
    result = pipe_xgb.predict(input_df)[0]
    st.success(f"🏏 Predicted Final Score: {int(result)} runs")

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Define teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 
         'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 
          'Cardiff', 'Christchurch', 'Trinidad']

st.title('T20 Cricket Score Predictor')

# Select boxes for teams and city
batting_team = st.selectbox('Select batting team', sorted(teams))
bowling_team = st.selectbox('Select bowling team', sorted(teams))
city = st.selectbox('Select city', sorted(cities))

# Number inputs for current score, overs, wickets, runs in last 5 overs
current_score = st.number_input('Current Score')
overs = st.number_input('Overs done (works for over > 5)', min_value=0.0, step=0.1)
wickets = st.number_input('Wickets out', min_value=0, step=1)
last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'crr': [crr],
        'last_five': [last_five]
    })

    # Encode categorical variables
    # Example assuming 'batting_team', 'bowling_team', 'city' are encoded with OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit_transform(input_df[['batting_team', 'bowling_team', 'city']])

    # Predict using the loaded pipeline
    result = pipe.predict(input_df)

    # Display predicted score
    st.header("Predicted Score - " + str(int(result[0])))


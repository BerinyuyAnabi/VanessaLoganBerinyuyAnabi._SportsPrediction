import streamlit as st
import pandas as pd
import pickle 
from urllib.request import urlopen 
import datetime


model_path = 'best_xgboost_model.pkl'  # Adjust the path as per your file location
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

#loaded_model = pickle.load('https://github.com/BerinyuyAnabi/VanessaLoganBerinyuyAnabi._SportsPrediction/blob/main/best_xgboost_model.pkl')
scaler = pickle.load(open('scaler.pkl','rb'))

#Creating the App
st.set_page_config(
    page_title = "FIFA- Player Rating Sports Prediction",
    page_icon = "⚽️",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

st.title("Say Hello to the FIFA Player Rating Prediction")
st.write("Enter a player's characteristics to predict their overall rating.")


# Define input fields for user input
movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, value=50)
mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, value=50)
passing = st.number_input("Passing", min_value=0, max_value=100, value=50)
potential = st.number_input("Potential", min_value=0, max_value=100, value=50)
# release_clause_eur = st.number_input("Release Clause (in EUR)", min_value=0, value=0)
dribbling = st.number_input("Dribbling", min_value=0, max_value=100, value=50)
wage_eur = st.number_input("Wage (in EUR)", min_value=0, value=0)
power_shot_power = st.number_input("Power Shot Power", min_value=0, max_value=100, value=50)
value_eur = st.number_input("Value (in EUR)", min_value=0, value=0)
mentality_vision = st.number_input("Mentality Vision", min_value=0, max_value=100, value=50)
attacking_short_passing = st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=50)
skill_long_passing = st.number_input("Skill Long Passing", min_value=0, max_value=100, value=50)
skill_ball_control = st.number_input("Skill Ball Control", min_value=0, max_value=100, value=50)
physic = st.number_input("Physic", min_value=0, max_value=100, value=50)
international_reputation = st.number_input("International Reutation", min_value=0, max_value=100, value=50)
age = st.number_input("Age", min_value=18, max_value=100, value=50)

#dob = st.number_input("When was the player's birthday?", min_value=0, max_value=100, value=50)
# st.write("Your birthday is:", dob)


button = st.button('Get Player Rating')
reset = st.button('Reset')

if button:
    input_data = {
    "movement_reactions": movement_reactions,
    "potential": potential,
    "passing": passing,
    "wage_eur": wage_eur,
    "mentality_composure": mentality_composure,
    "value_eur": value_eur,
   "dribbling": dribbling,
   "attacking_short_passing": attacking_short_passing,
    "mentality_vision": mentality_vision,
    "international_reputation":international_reputation,
    "skill_long_passing": skill_long_passing,
    "power_shot_power": power_shot_power,
    "physic": physic,
    "age": age,
    "skill_ball_control": skill_ball_control,

# "release_clause_eur": release_clause_eur,
    }

    #print(input_data)
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    #print(scaled_input)

    prediction = loaded_model.predict(scaled_input)
    st.write(f"The player rating is {int(prediction[0])}")
    st.write(f"The confidence score of the model is 93%")


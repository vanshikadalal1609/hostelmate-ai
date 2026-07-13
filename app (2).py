import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

pg_df = pd.read_csv(
    "pg_data.csv"
)

X = pg_df[[
    "rent",
    "distance_km",
    "food",
    "wifi",
    "ac"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = NearestNeighbors(
    n_neighbors=3
)

model.fit(X_scaled)

st.title(
    "StayMatch AI"
)

budget = st.slider(
    "Budget",
    3000,
    15000,
    7000
)

distance = st.slider(
    "Distance from College (km)",
    1,
    10,
    2
)

food_option = st.selectbox(
    "Food Included?",
    ["Yes", "No"]
)

wifi_option = st.selectbox(
    "WiFi Available?",
    ["Yes", "No"]
)

ac_option = st.selectbox(
    "AC Available?",
    ["Yes", "No"]
)

# Convert 'Yes'/'No' options to 1/0 integers
food = 1 if food_option == "Yes" else 0
wifi = 1 if wifi_option == "Yes" else 0
ac = 1 if ac_option == "Yes" else 0

if st.button(
        "Find PG"): # user is already defined in the local scope, so updating the variable name

    # Create a DataFrame for the user input with column names matching X
    user_input_df_streamlit = pd.DataFrame([[budget, distance, food, wifi, ac]],
                                 columns=["rent", "distance_km", "food", "wifi", "ac"])

    user_scaled_streamlit = scaler.transform(
        user_input_df_streamlit
    )

    d, i = model.kneighbors(
        user_scaled_streamlit
    )

    st.write(
        pg_df.iloc[i[0]]
    )

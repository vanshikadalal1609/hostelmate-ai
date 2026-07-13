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

food = st.selectbox(
    "Food Included?",
    [0,1]
)

wifi = st.selectbox(
    "WiFi?",
    [0,1]
)

ac = st.selectbox(
    "AC?",
    [0,1]
)

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

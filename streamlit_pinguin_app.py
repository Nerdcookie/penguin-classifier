
import streamlit as st
import pandas as pd
import joblib

# Modell laden
model = joblib.load("pinguin_klassifikator.pkl")

st.title("🐧 Pinguin-Klassifikation")
st.markdown("Gib die Merkmale eines Pinguins ein und erhalte eine Vorhersage der Art.")

# Eingabemaske
bill_length = st.number_input("Schnabellänge (mm)", value=40.0)
bill_depth = st.number_input("Schnabeltiefe (mm)", value=18.0)
flipper_length = st.number_input("Flipperlänge (mm)", value=200)
body_mass = st.number_input("Körpergewicht (g)", value=4200)

# Geschlecht (One-Hot-Encoding)
sex = st.selectbox("Geschlecht", ["male", "female"])
sex_male = 1 if sex == "male" else 0
sex_female = 1 if sex == "female" else 0

# Insel (One-Hot-Encoding)
island = st.selectbox("Insel", ["Biscoe", "Dream", "Torgersen"])
island_Biscoe = 1 if island == "Biscoe" else 0
island_Dream = 1 if island == "Dream" else 0
island_Torgersen = 1 if island == "Torgersen" else 0

# Vorhersage ausgeben
if st.button("Pinguin vorhersagen"):
    daten = [[bill_length, bill_depth, flipper_length, body_mass,
              sex_female, sex_male, island_Biscoe, island_Dream, island_Torgersen]]
    vorhersage = model.predict(daten)[0]
    st.success(f"👉 Vorhersage: **{vorhersage}**")

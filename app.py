import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# load our trained logistic model
with open ("model.pkl", "rb") as file:
    lr = pickle.load(file)

### Configuring Streamlit App ---

# Formatting
st.title("LinkedIn Prediction: Who's a User?")

# User Inputs
st.markdown("""
            ##### Household Income Level Legend
            * 1: Less than \$10,000
            * 2: \$10,000 to under \$20,000
            * 3: \$20,000 to under \$30,000
            * 4: \$30,000 to under \$40,000
            * 5: \$40,000 to under \$50,000
            * 6: \$50,000 to under \$75,000
            * 7: \$75,000 to under \$100,000
            * 8: \$100,000 to under \$150,000
            * 9: \$150,000 or more
            """)

income = st.number_input("**Household Income Level (1-9):**", min_value=1, max_value=9, step=1)

st.markdown("""
            ##### Education Level Legend
            * 1: Less than high school
            * 2: High school incomplete
            * 3: High school graduate
            * 4: Some college, no degree
            * 5: Two-year associate degree
            * 6: Four-year college or university degree/Bachelor's degree
            * 7: Some postgraduate school, no degree
            * 8: Postgraduate or professional degree
            """)

education = st.number_input("**Education Level (1-8):**", min_value=1, max_value=8, step=1)
parent = st.selectbox("**Are you a parent?**", ["Yes", "No"])
married = st.selectbox("**Marital Status**", ["Married", "Not Married"])
female = st.selectbox("**Gender**", ["Female", "Male"])
age = st.number_input("**Age:**", min_value=18, max_value=100, step=1)

# Convert categorical inputs to binary
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Married" else 0
female = 1 if female == "Female" else 0

# Predict LinkedIn usage
if st.button("Predict"):
    features = np.array([[income, education, parent, married, female, age]])
    prediction = lr.predict(features)[0]
    probability = lr.predict_proba(features)[0][1]

    st.write("## Prediction:")
    st.write("##### Uses LinkedIn :white_check_mark:" if prediction == 1 else "##### Does Not Use LinkedIn :x:")
    
    st.write("## Probability of Using LinkedIn:")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value = probability * 100,
        number={'suffix': '%', 'font': {'color': "#2D2D60"}},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#2D2D60"}}))
    st.plotly_chart(fig)

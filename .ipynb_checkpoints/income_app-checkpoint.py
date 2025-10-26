import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("income_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Income Prediction App", layout="centered")

st.title("💰 Income Classification App")
st.write(
    "Predict whether a person earns more than **$50K per year** based on demographic information."
)


st.sidebar.header("🧮 Input Features")

# --- Numerical inputs ---
age = st.sidebar.slider("Age", 17, 90, 35)
education_num = st.sidebar.slider("Education Number (Years)", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
capital_gain = st.sidebar.number_input(
    "Capital Gain", min_value=0, max_value=100000, value=0
)
capital_loss = st.sidebar.number_input(
    "Capital Loss", min_value=0, max_value=10000, value=0
)

# --- Categorical selections ---
workclass = st.sidebar.selectbox(
    "Workclass",
    [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"],
)

occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
        "None",
    ],
)

race = st.sidebar.selectbox(
    "Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
)

sex = st.sidebar.radio("Sex", ["Female", "Male"])

native_country = st.sidebar.selectbox(
    "Native Country",
    [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Canada",
        "India",
        "England",
        "Cuba",
        "Jamaica",
        "China",
        "Italy",
        "Puerto-Rico",
        "Japan",
        "South",
        "Other",
    ],
)


input_data = pd.DataFrame(
    {
        "age": [age],
        "workclass": [workclass],
        "education-num": [education_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "race": [race],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country],
        "sex_Male": [1 if sex == "Male" else 0],
    }
)


# Label encoding for other categorical columns
from sklearn.preprocessing import LabelEncoder

categorical_cols = [
    "workclass",
    "marital-status",
    "occupation",
    "race",
    "native-country",
]
le = LabelEncoder()

# Ensure same mapping as training
for col in categorical_cols:
    input_data[col] = le.fit_transform(input_data[col])

# Scale numeric features
scaled_features = scaler.transform(input_data)


if st.button("Predict Income"):
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    if prediction == 1:
        st.success(f"✅ Predicted: Income > 50K (Confidence: {probability:.2%})")
    else:
        st.warning(f"⚠️ Predicted: Income ≤ 50K (Confidence: {1 - probability:.2%})")

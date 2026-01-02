import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Electricity Consumption Prediction ⚡",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("⚡ Electricity Consumption Prediction App")
    st.markdown("""
    ### Welcome 👋  

    This is a **Machine Learning based Streamlit application**  
    to predict **monthly electricity consumption**.

    ### Features used:
    - Appliance type & power
    - Household details
    - Season & environment
    - Behavioral patterns

    👉 Use the **sidebar** to go to the Prediction page.
    """)
    st.success("Select a page from the sidebar ⬅️")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":
    st.title("🔮 Monthly Electricity Consumption Prediction")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    data = pd.read_csv("electricity_consumption_monthly.csv")

    # Encode categorical columns
    encoder_building = LabelEncoder()
    data["building_type"] = encoder_building.fit_transform(data["building_type"])
    encoder_season = LabelEncoder()
    data["season"] = encoder_season.fit_transform(data["season"])

    # Features and target
    features = ["occupants", "building_type", "season", "temperature_c", "vacation_days"]
    X = data[features]
    y = data["monthly_electricity_units"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    bagging = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)

    lr.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    bagging.fit(X_train, y_train)

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    st.subheader("Enter Household & Usage Details")

    occupants = st.number_input("Number of Occupants", 1, 10)
    building_type = st.selectbox("Building Type", ["Apartment", "Independent"])
    season = st.selectbox("Season", ["Summer", "Winter", "Rainy"])
    temperature = st.slider("Temperature (°C)", 10, 50)
    vacation = st.number_input("Vacation Days (per month)", 0, 15)

    # Encode inputs
    building_type_enc = 0 if building_type == "Apartment" else 1
    season_enc = {"Summer": 0, "Winter": 1, "Rainy": 2}[season]

    # -----------------------------
    # PREDICTION BUTTON
    # -----------------------------
    if st.button("⚡ Predict Electricity Consumption"):
        input_data = np.array([[occupants, building_type_enc, season_enc, temperature, vacation]])

        lr_pred = lr.predict(input_data)[0]
        ridge_pred = ridge.predict(input_data)[0]
        bagging_pred = bagging.predict(input_data)[0]

        ensemble_pred = (lr_pred + ridge_pred + bagging_pred) / 3

        st.divider()
        st.success(f"🔌 Estimated Monthly Consumption: **{ensemble_pred:.2f} units**")

        st.write("### Model-wise Predictions")
        st.write(f"• Linear Regression: {lr_pred:.2f} units")
        st.write(f"• Ridge Regression: {ridge_pred:.2f} units")
        st.write(f"• Bagging Regression: {bagging_pred:.2f} units")

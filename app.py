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

    This **Machine Learning based Streamlit application** predicts  
    **monthly electricity consumption** using household and
    environmental data.

    ### Models Used
    - Linear Regression
    - Ridge Regression
    - Bagging Regression (Ensemble)

    👉 Use the **sidebar** to navigate to the Prediction page.
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

    # -----------------------------
    # ENCODING
    # -----------------------------
    encoder_building = LabelEncoder()
    encoder_season = LabelEncoder()

    data["building_type"] = encoder_building.fit_transform(data["building_type"])
    data["season"] = encoder_season.fit_transform(data["season"])

    # -----------------------------
    # FEATURES & TARGET
    # -----------------------------
    features = [
        "occupants",
        "building_type",
        "season",
        "temperature_c",
        "vacation_days"
    ]

    X = data[features]
    y = data["monthly_electricity_units"]

    # -----------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    lr_model = LinearRegression()
    ridge_model = Ridge(alpha=1.0)
    bagging_model = BaggingRegressor(
        estimator=LinearRegression(),
        n_estimators=10,
        random_state=42
    )

    lr_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    bagging_model.fit(X_train, y_train)

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    st.subheader("🏠 Enter Household & Usage Details")

    occupants = st.number_input(
        "Number of Occupants",
        min_value=1,
        max_value=10,
        value=3
    )

    building_type = st.selectbox(
        "Building Type",
        encoder_building.classes_
    )

    season = st.selectbox(
        "Season",
        encoder_season.classes_
    )

    temperature = st.slider(
        "Temperature (°C)",
        min_value=10,
        max_value=50,
        value=25
    )

    vacation_days = st.number_input(
        "Vacation Days (per month)",
        min_value=0,
        max_value=15,
        value=2
    )

    # -----------------------------
    # ENCODE USER INPUT
    # -----------------------------
    building_type_enc = encoder_building.transform([building_type])[0]
    season_enc = encoder_season.transform([season])[0]

    input_data = np.array([[
        occupants,
        building_type_enc,
        season_enc,
        temperature,
        vacation_days
    ]])

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if st.button("⚡ Predict Electricity Consumption"):

        lr_pred = lr_model.predict(input_data)[0]
        ridge_pred = ridge_model.predict(input_data)[0]
        bagging_pred = bagging_model.predict(input_data)[0]

        ensemble_pred = (lr_pred + ridge_pred + bagging_pred) / 3

        st.divider()
        st.success(
            f"🔌 Estimated Monthly Electricity Consumption: "
            f"**{ensemble_pred:.2f} units**"
        )

        st.write("### 🔍 Model-wise Predictions")
        st.write(f"• Linear Regression: **{lr_pred:.2f} units**")
        st.write(f"• Ridge Regression: **{ridge_pred:.2f} units**")
        st.write(f"• Bagging Regression: **{bagging_pred:.2f} units**")

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model Bundle
# -----------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("Toolkit/ChurnPredictor.joblib")

bundle = load_bundle()

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to **churn (leave)** or **stay**."
)

# -----------------------------
# Customer Information
# -----------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "No internet service", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "No internet service", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "No internet service", "Yes"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "No internet service", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "No internet service", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "No internet service", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

# -----------------------------
# Numeric Features
# -----------------------------
st.header("Account Charges")

col3, col4, col5 = st.columns(3)

with col3:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col4:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

with col5:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=1.0
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Churn", type="primary"):

    # Create dataframe with all required model columns
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    # Binary Features
    input_df[f"SeniorCitizen_{senior_citizen}"] = 1
    input_df[f"Partner_{partner}"] = 1
    input_df[f"Dependents_{dependents}"] = 1

    # Internet Features
    input_df[f"InternetService_{internet_service}"] = 1
    input_df[f"OnlineSecurity_{online_security}"] = 1
    input_df[f"OnlineBackup_{online_backup}"] = 1
    input_df[f"DeviceProtection_{device_protection}"] = 1
    input_df[f"TechSupport_{tech_support}"] = 1
    input_df[f"StreamingTV_{streaming_tv}"] = 1
    input_df[f"StreamingMovies_{streaming_movies}"] = 1

    # Contract & Billing
    input_df[f"Contract_{contract}"] = 1
    input_df[f"PaperlessBilling_{paperless_billing}"] = 1
    input_df[f"PaymentMethod_{payment_method}"] = 1

    # Numeric Features
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges

    # Scale numeric columns
    input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    stay_probability = probabilities[0]
    churn_probability = probabilities[1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to churn\n\n"
            f"Confidence: {churn_probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay\n\n"
            f"Confidence: {stay_probability:.2%}"
        )

    st.subheader("Probability Breakdown")

    results = pd.DataFrame({
        "Outcome": ["Stay", "Churn"],
        "Probability": [
            round(stay_probability * 100, 2),
            round(churn_probability * 100, 2)
        ]
    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("Show Model Input"):
        st.dataframe(input_df)
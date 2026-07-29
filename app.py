import sys
import os

# --- AWS ENVIRONMENT BINDING INJECTION ---
# Sets environment variables BEFORE Streamlit imports/initializes server config
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INLINE CSS & FLYING BUTTERFLIES ANIMATION ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background Gradient: Sky Blue to Soft Sunset Orange */
.stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 35%, #685290 70%, #f79d65 100%);
    background-attachment: fixed;
    color: #ffffff;
}

/* Glassmorphism Containers */
div[data-testid="stForm"], div.stMarkdownContainer, div[data-testid="stMetricValue"] {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

/* Header Banner */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 0 15px rgba(255, 215, 0, 0.5), 0 0 25px rgba(255, 154, 118, 0.7);
    letter-spacing: 1px;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #f0e6ff;
    margin-top: 5px;
}

/* Submit Button Styling */
.stButton>button {
    width: 100%;
    background: linear-gradient(45deg, #ff7e5f, #feb47b);
    color: white !important;
    font-weight: 600;
    font-size: 1.1rem;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 126, 95, 0.4);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 126, 95, 0.7);
    background: linear-gradient(45deg, #feb47b, #ff7e5f);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* CONTINUOUS FLYING BUTTERFLIES ANIMATION */
.butterfly-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}

.butterfly {
    position: absolute;
    width: 35px;
    height: 35px;
    opacity: 0.85;
    animation: fly 14s infinite linear;
    filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.8));
}

.bf1 { top: 90%; left: -5%; animation-duration: 12s; animation-delay: 0s; }
.bf2 { top: 75%; left: -5%; animation-duration: 16s; animation-delay: 3s; }
.bf3 { top: 60%; left: -5%; animation-duration: 14s; animation-delay: 6s; }
.bf4 { top: 85%; left: -5%; animation-duration: 18s; animation-delay: 9s; }
.bf5 { top: 50%; left: -5%; animation-duration: 13s; animation-delay: 4.5s; }

@keyframes fly {
    0% {
        transform: translate(0, 0) rotate(15deg) scale(0.8);
        opacity: 0;
    }
    10% {
        opacity: 0.9;
    }
    50% {
        transform: translate(50vw, -40vh) rotate(-10deg) scale(1.1);
    }
    90% {
        opacity: 0.9;
    }
    100% {
        transform: translate(105vw, -90vh) rotate(20deg) scale(0.8);
        opacity: 0;
    }
}
</style>

<!-- FLYING BUTTERFLIES OVERLAY -->
<div class="butterfly-container">
    <svg class="butterfly bf1" viewBox="0 0 50 50">
        <path fill="#ffd166" d="M25,25 Q10,5 5,20 Q10,35 25,25 Q40,5 45,20 Q40,35 25,25 Z"/>
        <path fill="#ff9f1c" d="M25,25 Q15,38 8,45 Q20,48 25,25 Q35,38 42,45 Q30,48 25,25 Z"/>
    </svg>
    <svg class="butterfly bf2" viewBox="0 0 50 50">
        <path fill="#4cc9f0" d="M25,25 Q10,5 5,20 Q10,35 25,25 Q40,5 45,20 Q40,35 25,25 Z"/>
        <path fill="#4895ef" d="M25,25 Q15,38 8,45 Q20,48 25,25 Q35,38 42,45 Q30,48 25,25 Z"/>
    </svg>
    <svg class="butterfly bf3" viewBox="0 0 50 50">
        <path fill="#f72585" d="M25,25 Q10,5 5,20 Q10,35 25,25 Q40,5 45,20 Q40,35 25,25 Z"/>
        <path fill="#b5179e" d="M25,25 Q15,38 8,45 Q20,48 25,25 Q35,38 42,45 Q30,48 25,25 Z"/>
    </svg>
    <svg class="butterfly bf4" viewBox="0 0 50 50">
        <path fill="#ffffff" d="M25,25 Q10,5 5,20 Q10,35 25,25 Q40,5 45,20 Q40,35 25,25 Z"/>
        <path fill="#ffd166" d="M25,25 Q15,38 8,45 Q20,48 25,25 Q35,38 42,45 Q30,48 25,25 Z"/>
    </svg>
    <svg class="butterfly bf5" viewBox="0 0 50 50">
        <path fill="#ff9a76" d="M25,25 Q10,5 5,20 Q10,35 25,25 Q40,5 45,20 Q40,35 25,25 Z"/>
        <path fill="#ffc857" d="M25,25 Q15,38 8,45 Q20,48 25,25 Q35,38 42,45 Q30,48 25,25 Z"/>
    </svg>
</div>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- AWS SAFE RELATIVE PATH LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Customer_churn.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file `{MODEL_PATH}` was not found in the root directory.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model file: {e}")
        return None

model = load_model()

# --- HEADER BANNER ---
st.markdown("""
<div class="main-header">
    <div class="main-title">Customer Churn Predictor</div>
    <div class="main-subtitle">Analyze customer usage metrics to predict potential churn risk</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
st.sidebar.title("Navigation & Status")
st.sidebar.info("Provide customer details in the main screen and click predict to calculate churn probability.")

if model is not None:
    st.sidebar.success("Model status: Loaded & Ready")

# --- USER INPUT FORM ---
st.markdown("### Customer Information Inputs")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", options=["Female", "Male"])
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=120, value=24, step=1)
        usage_freq = st.number_input("Usage Frequency (per month)", min_value=1, max_value=30, value=15, step=1)

    with col2:
        support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2, step=1)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=5, step=1)
        sub_type = st.selectbox("Subscription Type", options=["Basic", "Standard", "Premium"])

    with col3:
        contract_len = st.selectbox("Contract Length", options=["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=90, value=10, step=1)

    submit_button = st.form_submit_button(label="Predict Churn Risk")

# --- CATEGORICAL ENCODING MAPPINGS ---
gender_map = {"Female": 0, "Male": 1}
sub_map = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

# --- PREDICTION ENGINE ---
if submit_button:
    if model is None:
        st.error("Cannot process prediction because the model file is unavailable.")
    else:
        try:
            # Features match exact order in Customer_churn.pkl AdaBoostClassifier
            input_data = pd.DataFrame([{
                'Age': age,
                'Gender': gender_map[gender],
                'Tenure': tenure,
                'Usage Frequency': usage_freq,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Subscription Type': sub_map[sub_type],
                'Contract Length': contract_map[contract_len],
                'Total Spend': total_spend,
                'Last Interaction': last_interaction
            }])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.markdown("### Prediction Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prediction == 1:
                    st.error("⚠️ **High Churn Risk Detected**")
                else:
                    st.success("✅ **Low Churn Risk (Customer Retained)**")

            with res_col2:
                if probability is not None:
                    churn_prob = probability[1] * 100
                    st.metric(label="Calculated Churn Probability", value=f"{churn_prob:.1f}%")

        except Exception as err:
            st.error(f"An error occurred during prediction processing: {err}")

# --- AUTOMATED ENTRYPOINT LAUNCHER FOR AWS CONTAINER ENVIRONMENT ---
if __name__ == "__main__":
    # Allows running directly with `python app.py` on AWS EC2 or Elastic Beanstalk
    from streamlit.web import cli as stcli
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

import os
import joblib
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# --- SAFE RELATIVE PATH LOADING FOR AWS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Customer_churn.pkl")

# Load model on startup
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

# Feature Mappings
GENDER_MAP = {"Female": 0, "Male": 1}
SUB_MAP = {"Basic": 0, "Standard": 1, "Premium": 2}
CONTRACT_MAP = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

# --- EMBEDDED HTML, CSS & ANIMATION TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 35%, #685290 70%, #f79d65 100%);
            background-attachment: fixed;
            color: #ffffff;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            padding: 2rem 1rem;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .title {
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
        }

        .subtitle {
            font-size: 1rem;
            color: #f0e6ff;
            margin-top: 5px;
        }

        .card {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-size: 0.9rem;
            font-weight: 600;
        }

        input, select {
            width: 100%;
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.2);
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
        }

        select option {
            background: #2a5298;
            color: #fff;
        }

        button {
            width: 100%;
            margin-top: 25px;
            background: linear-gradient(45deg, #ff7e5f, #feb47b);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            border: none;
            border-radius: 12px;
            padding: 14px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 126, 95, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 126, 95, 0.7);
        }

        .alert {
            padding: 15px 20px;
            border-radius: 12px;
            margin-top: 20px;
            font-weight: 600;
        }

        .alert-error {
            background: rgba(239, 68, 68, 0.3);
            border: 1px solid #ef4444;
        }

        .alert-success {
            background: rgba(34, 197, 94, 0.3);
            border: 1px solid #22c55e;
        }

        /* FLYING BUTTERFLIES ANIMATION */
        .butterfly-container {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            pointer-events: none;
            z-index: 9999;
            overflow: hidden;
        }

        .butterfly {
            position: absolute;
            width: 35px; height: 35px;
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
            0% { transform: translate(0, 0) rotate(15deg) scale(0.8); opacity: 0; }
            10% { opacity: 0.9; }
            50% { transform: translate(50vw, -40vh) rotate(-10deg) scale(1.1); }
            90% { opacity: 0.9; }
            100% { transform: translate(105vw, -90vh) rotate(20deg) scale(0.8); opacity: 0; }
        }
    </style>
</head>
<body>

    <!-- FLYING BUTTERFLIES -->
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

    <div class="container">
        <div class="header">
            <div class="title">🦋 Customer Churn Predictor</div>
            <div class="subtitle">Analyze customer usage metrics to predict potential churn risk</div>
        </div>

        {% if error %}
        <div class="alert alert-error">⚠️ {{ error }}</div>
        {% endif %}

        <div class="card">
            <h3 style="margin-bottom: 20px;">Customer Information Inputs</h3>
            <form method="POST">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="age" value="35" min="18" max="100" required>
                    </div>
                    <div class="form-group">
                        <label>Gender</label>
                        <select name="gender">
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Tenure (months)</label>
                        <input type="number" name="tenure" value="24" min="1" max="120" required>
                    </div>
                    <div class="form-group">
                        <label>Usage Frequency (per month)</label>
                        <input type="number" name="usage_freq" value="15" min="1" max="30" required>
                    </div>
                    <div class="form-group">
                        <label>Support Calls</label>
                        <input type="number" name="support_calls" value="2" min="0" max="20" required>
                    </div>
                    <div class="form-group">
                        <label>Payment Delay (days)</label>
                        <input type="number" name="payment_delay" value="5" min="0" max="60" required>
                    </div>
                    <div class="form-group">
                        <label>Subscription Type</label>
                        <select name="sub_type">
                            <option value="Basic">Basic</option>
                            <option value="Standard">Standard</option>
                            <option value="Premium">Premium</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Contract Length</label>
                        <select name="contract_len">
                            <option value="Monthly">Monthly</option>
                            <option value="Quarterly">Quarterly</option>
                            <option value="Annual">Annual</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Total Spend ($)</label>
                        <input type="number" step="0.1" name="total_spend" value="500" min="0" required>
                    </div>
                    <div class="form-group">
                        <label>Last Interaction (days ago)</label>
                        <input type="number" name="last_interaction" value="10" min="0" max="90" required>
                    </div>
                </div>

                <button type="submit">Predict Churn Risk</button>
            </form>
        </div>

        {% if prediction is not none %}
        <div class="card">
            <h3>Prediction Results</h3>
            {% if prediction == 1 %}
                <div class="alert alert-error">⚠️ <b>High Churn Risk Detected</b></div>
            {% else %}
                <div class="alert alert-success">✅ <b>Low Churn Risk (Customer Retained)</b></div>
            {% endif %}

            {% if probability is not none %}
                <p style="margin-top: 15px; font-size: 1.1rem;">
                    Calculated Churn Probability: <b>{{ probability }}%</b>
                </p>
            {% endif %}
        </div>
        {% endif %}
    </div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    probability = None
    error_message = None

    if request.method == "POST":
        if model is None:
            error_message = "Model file 'Customer_churn.pkl' was not found or could not be loaded."
        else:
            try:
                # Get inputs from form
                age = float(request.form.get("age", 35))
                gender = request.form.get("gender", "Female")
                tenure = float(request.form.get("tenure", 24))
                usage_freq = float(request.form.get("usage_freq", 15))
                support_calls = float(request.form.get("support_calls", 2))
                payment_delay = float(request.form.get("payment_delay", 5))
                sub_type = request.form.get("sub_type", "Basic")
                contract_len = request.form.get("contract_len", "Monthly")
                total_spend = float(request.form.get("total_spend", 500))
                last_interaction = float(request.form.get("last_interaction", 10))

                # Structure DataFrame matching AdaBoost model feature order:
                # ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction']
                input_data = pd.DataFrame([{
                    'Age': age,
                    'Gender': GENDER_MAP.get(gender, 0),
                    'Tenure': tenure,
                    'Usage Frequency': usage_freq,
                    'Support Calls': support_calls,
                    'Payment Delay': payment_delay,
                    'Subscription Type': SUB_MAP.get(sub_type, 0),
                    'Contract Length': CONTRACT_MAP.get(contract_len, 0),
                    'Total Spend': total_spend,
                    'Last Interaction': last_interaction
                }])

                # Run prediction
                pred = model.predict(input_data)[0]
                prediction_result = int(pred)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)[0]
                    probability = round(probs[1] * 100, 1)

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction_result,
        probability=probability,
        error=error_message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)

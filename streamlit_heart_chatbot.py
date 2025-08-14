import streamlit as st
from streamlit_chat import message
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
CSV_FILE = "prediction_history.csv"

# SHAP explainer for XGBoost
explainer = shap.TreeExplainer(model)

# Save prediction (kept for internal logging; no public download button)
def save_prediction(data, prediction):
    data["prediction (%)"] = round(prediction, 2)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)

# Generate PDF report
def generate_pdf(input_data, prediction):
    field_names = {
        "age": "Age",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mm Hg)",
        "chol": "Cholesterol Level (mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
        "restecg": "Resting ECG Result (0–2)",
        "thalach": "Max Heart Rate Achieved",
        "exang": "Exercise-Induced Angina (0=No, 1=Yes)",
        "oldpeak": "ST Depression by Exercise",
        "slope": "Slope of Peak Exercise ST",
        "ca": "Major Vessels Coloured (0–4)",
        "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
    }
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    text.textLine("Heart Disease Risk Assessment Report")
    text.textLine("--------------------------------------")
    for key, value in input_data.items():
        label = field_names.get(key, key)
        text.textLine(f"{label}: {value}")
    text.textLine(f"\nPredicted Risk: {round(prediction, 2)}%")
    text.textLine(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Constraints for validation
CONSTRAINTS = {
    "age":      {"type": int,   "min": 18,  "max": 100},
    "sex":      {"type": int,   "choices": [0, 1]},
    "cp":       {"type": int,   "min": 0,   "max": 3},
    "trestbps": {"type": int,   "min": 80,  "max": 220},
    "chol":     {"type": int,   "min": 100, "max": 700},
    "fbs":      {"type": int,   "choices": [0, 1]},
    "restecg":  {"type": int,   "choices": [0, 1, 2]},
    "thalach":  {"type": int,   "min": 60,  "max": 230},
    "exang":    {"type": int,   "choices": [0, 1]},
    "oldpeak":  {"type": float, "min": 0.0, "max": 7.0},
    "slope":    {"type": int,   "choices": [0, 1, 2]},
    "ca":       {"type": int,   "min": 0,   "max": 4},
    "thal":     {"type": int,   "choices": [0, 1, 2]},
}

# Questions with integrated valid ranges/choices
questions = [
    {"key": "age",      "text": "What is your age (18–100)?",                              "type": int},
    {"key": "sex",      "text": "What is your biological sex (0 = Female, 1 = Male)?",     "type": int},
    {"key": "cp",       "text": "What is your chest pain type (0–3)?",                     "type": int},
    {"key": "trestbps", "text": "Resting blood pressure (80–220 mm Hg)?",                  "type": int},
    {"key": "chol",     "text": "Cholesterol level (100–700 mg/dl)?",                      "type": int},
    {"key": "fbs",      "text": "Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes)?",      "type": int},
    {"key": "restecg",  "text": "Resting ECG result (0–2)?",                               "type": int},
    {"key": "thalach",  "text": "Maximum heart rate achieved (60–230)?",                   "type": int},
    {"key": "exang",    "text": "Exercise-induced angina (0 = No, 1 = Yes)?",              "type": int},
    {"key": "oldpeak",  "text": "ST depression induced by exercise (0.0–7.0)?",            "type": float},
    {"key": "slope",    "text": "Slope of peak ST segment (0–2)?",                         "type": int},
    {"key": "ca",       "text": "Number of major vessels coloured (0–4)?",                 "type": int},
    {"key": "thal",     "text": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)?",          "type": int}
]

def coerce_and_validate(key: str, raw_text: str):
    """Cast input to the right dtype and enforce feasible ranges/choices."""
    spec = CONSTRAINTS[key]
    caster = spec["type"]
    val = caster(raw_text)
    if "choices" in spec:
        if val not in spec["choices"]:
            raise ValueError(f"Value must be one of {spec['choices']}.")
    else:
        if ("min" in spec and val < spec["min"]) or ("max" in spec and val > spec["max"]):
            lo = spec.get("min", "-∞")
            hi = spec.get("max", "+∞")
            raise ValueError(f"Value must be between {lo} and {hi}.")
    return val

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# Title
st.title("💬 Heart Disease Risk Chatbot (Chat Mode)")

# Display chat
for i, msg in enumerate(st.session_state.chat_history):
    message(msg["text"], is_user=msg["is_user"], key=f"{'user' if msg['is_user'] else 'bot'}-{i}")

# Ask questions
if st.session_state.current_q < len(questions):
    q = questions[st.session_state.current_q]
    user_input = st.chat_input(q["text"])
    if user_input:
        try:
            typed_input = coerce_and_validate(q["key"], user_input)
            st.session_state.answers[q["key"]] = typed_input
            st.session_state.chat_history.append({"text": q["text"], "is_user": False})
            st.session_state.chat_history.append({"text": user_input, "is_user": True})
            st.session_state.current_q += 1
            st.rerun()
        except ValueError as e:
            st.warning(f"Please enter a valid value. {e}")

# After final answer
else:
    inputs = [st.session_state.answers[q["key"]] for q in questions]
    input_array = scaler.transform([inputs])
    prediction = model.predict_proba(input_array)[0][1] * 100
    save_prediction(st.session_state.answers, prediction)

    st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
    if prediction > 70:
        st.warning("⚠️ High risk! Please consult a doctor.")
    elif prediction > 40:
        st.info("🔍 Moderate risk. A check-up is recommended.")
    else:
        st.info("✅ Low risk. Great job keeping healthy!")

    # Pie chart
    st.markdown("### 📊 Risk Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie([prediction, 100 - prediction], labels=["At Risk", "No Risk"],
            colors=["red", "green"], autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    # SHAP bar plot
    st.markdown("### 🧠 Feature Contributions (Explainability)")
    shap_values = explainer.shap_values(input_array)
    shap_df = pd.DataFrame({
        "feature": [q["key"] for q in questions],
        "value": input_array[0],
        "shap": shap_values[0]
    }).sort_values("shap", key=abs, ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(shap_df["feature"], shap_df["shap"],
             color=["red" if x > 0 else "blue" for x in shap_df["shap"]])
    ax2.set_xlabel("SHAP Value (Impact on Prediction)")
    ax2.set_title("Top Feature Influences on Risk")
    st.pyplot(fig2)

    # PDF download only (removed "Download All Predictions")
    pdf = generate_pdf(st.session_state.answers, prediction)
    st.download_button("📄 Download PDF Report", data=pdf,
                       file_name="heart_risk_report.pdf", mime="application/pdf")

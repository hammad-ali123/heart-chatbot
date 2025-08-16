import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_chat import message
import shap
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Heart Risk Chatbot (Explainable)", page_icon="❤️", layout="centered")

# -----------------------------
# Load model & scaler (cache OK)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = "model.pkl"      # keep your original filenames
    scaler_path = "scaler.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file '{scaler_path}' not found.")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Feature names (codes -> full)
# -----------------------------
FIELD_NAMES = {
    "age": "Age (years)",
    "sex": "Sex (0=Female, 1=Male)",
    "cp": "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (0=No, 1=Yes)",
    "restecg": "Resting ECG (0–2)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (0=No, 1=Yes)",
    "oldpeak": "ST Depression (exercise vs rest)",
    "slope": "Slope of Peak Exercise ST (0–2)",
    "ca": "Major Vessels Colored by Fluoroscopy (0–3)",
    "thal": "Thalassemia Code (0–3)"
}

# -----------------------------
# Feasible ranges / choices
# -----------------------------
CONSTRAINTS = {
    "age": {"type": int, "min": 18, "max": 100},
    "sex": {"type": int, "choices": [0, 1]},
    "cp": {"type": int, "choices": [0, 1, 2, 3]},
    "trestbps": {"type": int, "min": 80, "max": 220},
    "chol": {"type": int, "min": 100, "max": 600},
    "fbs": {"type": int, "choices": [0, 1]},
    "restecg": {"type": int, "choices": [0, 1, 2]},
    "thalach": {"type": int, "min": 60, "max": 250},
    "exang": {"type": int, "choices": [0, 1]},
    "oldpeak": {"type": float, "min": 0.0, "max": 6.5},
    "slope": {"type": int, "choices": [0, 1, 2]},
    "ca": {"type": int, "choices": [0, 1, 2, 3]},
    "thal": {"type": int, "choices": [0, 1, 2, 3]},  # expanded to 0–3
}

# -----------------------------
# Chat questions (order = model features order)
# -----------------------------
QUESTIONS = [
    {"key": "age", "text": "Enter your age (valid 18–100):", "type": int},
    {"key": "sex", "text": "Sex (0 = Female, 1 = Male):", "type": int},
    {"key": "cp", "text": "Chest pain type (0–3 as per dataset encoding):", "type": int},
    {"key": "trestbps", "text": "Resting blood pressure (mm Hg, valid 80–220):", "type": int},
    {"key": "chol", "text": "Serum cholesterol (mg/dl, valid 100–600):", "type": int},
    {"key": "fbs", "text": "Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes):", "type": int},
    {"key": "restecg", "text": "Resting electrocardiographic results (0–2):", "type": int},
    {"key": "thalach", "text": "Maximum heart rate achieved (valid 60–250):", "type": int},
    {"key": "exang", "text": "Exercise induced angina (0 = No, 1 = Yes):", "type": int},
    {"key": "oldpeak", "text": "ST depression induced by exercise relative to rest (valid 0.0–6.5):", "type": float},
    {"key": "slope", "text": "Slope of peak exercise ST segment (0–2):", "type": int},
    {"key": "ca", "text": "Number of major vessels colored by fluoroscopy (0–3):", "type": int},
    {"key": "thal", "text": "Thalassemia (numeric code 0–3 as per dataset encoding):", "type": int},
]

# -----------------------------
# Helpers
# -----------------------------
def validate_and_cast(key: str, raw_val: str):
    meta = CONSTRAINTS[key]
    typ = meta["type"]
    try:
        val = typ(raw_val)
    except Exception:
        raise ValueError(f"'{FIELD_NAMES.get(key, key)}' must be of type {typ.__name__}.")
    if "choices" in meta:
        if val not in meta["choices"]:
            raise ValueError(f"'{FIELD_NAMES.get(key, key)}' must be one of {meta['choices']}.")
    else:
        lo, hi = meta["min"], meta["max"]
        if not (lo <= val <= hi):
            raise ValueError(f"'{FIELD_NAMES.get(key, key)}' must be between {lo} and {hi}.")
    return val

def get_explainer(trained_model):
    # No caching here (hashing tree models/XGB can fail). This is lightweight for single prediction.
    try:
        return shap.TreeExplainer(trained_model)
    except Exception:
        return shap.Explainer(trained_model)

def shap_positive_class_vector(explainer, model, input_row_2d: np.ndarray) -> np.ndarray:
    raw = explainer.shap_values(input_row_2d)
    # SHAP can return list [neg, pos] for tree-based binary classifiers
    if isinstance(raw, list):
        pos_idx = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
        vec = raw[pos_idx][0]
    else:
        vec = raw[0] if getattr(raw, "ndim", 1) > 1 else raw
    return np.asarray(vec).ravel()

def build_pdf_bytes(input_dict: dict, prediction_pct: float, shap_vec: np.ndarray, features_order: list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Heart Risk Prediction Report</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Predicted risk (class 1 probability): <b>{prediction_pct:.1f}%</b>", styles["Normal"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Inputs</b>", styles["Heading3"]))
    inputs_data = [["Code", "Feature", "Value"]]
    for f in features_order:
        inputs_data.append([f, FIELD_NAMES.get(f, f), str(input_dict.get(f, ""))])
    inputs_tbl = Table(inputs_data, hAlign="LEFT")
    inputs_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    story.append(inputs_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Local Feature Contributions (SHAP)</b>", styles["Heading3"]))
    shap_df = pd.DataFrame({
        "Code": features_order,
        "Feature": [FIELD_NAMES.get(f, f) for f in features_order],
        "Value": [input_dict.get(f, "") for f in features_order],
        "SHAP": shap_vec
    })
    shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index)

    contrib_data = [["Code", "Feature", "Value", "SHAP (impact on risk)"]]
    for _, r in shap_df.iterrows():
        contrib_data.append([r["Code"], r["Feature"], str(r["Value"]), f"{r['SHAP']:.4f}"])

    contrib_tbl = Table(contrib_data, hAlign="LEFT")
    contrib_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    story.append(contrib_tbl)

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# -----------------------------
# Session state for chat
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "completed" not in st.session_state:
    st.session_state.completed = False

# -----------------------------
# Header & helper expander
# -----------------------------
st.title("❤️ Explainable Heart Risk Chatbot")

with st.expander("Valid input ranges & choices"):
    txt = []
    for k, meta in CONSTRAINTS.items():
        if "choices" in meta:
            txt.append(f"- **{k}**: choices {meta['choices']}")
        else:
            txt.append(f"- **{k}**: {meta['min']} to {meta['max']}")
    st.markdown("\n".join(txt))

st.caption("This tool estimates cardiovascular disease risk with a machine-learning model and explains the factors that influenced the prediction. It is **not** a medical diagnosis.")

# -----------------------------
# Chat Display
# -----------------------------
for i, m in enumerate(st.session_state.messages):
    message(m["content"], is_user=m["is_user"], key=f"msg_{i}")

def ask_current_question():
    qi = st.session_state.q_index
    if qi < len(QUESTIONS):
        q = QUESTIONS[qi]
        prompt = QUESTIONS[qi]["text"]
        message(prompt, is_user=False, key=f"bot_q_{qi}")

def process_user_reply(reply_text: str):
    qi = st.session_state.q_index
    q = QUESTIONS[qi]
    key = q["key"]
    try:
        val = validate_and_cast(key, reply_text)
    except Exception as e:
        # Bot warns and re-asks same question
        st.session_state.messages.append({"content": reply_text, "is_user": True})
        st.session_state.messages.append({"content": f"⚠️ {e} Please try again.", "is_user": False})
        return

    # Save valid answer
    st.session_state.answers[key] = val
    st.session_state.messages.append({"content": reply_text, "is_user": True})

    # Move to next question or finish
    st.session_state.q_index += 1
    if st.session_state.q_index >= len(QUESTIONS):
        st.session_state.completed = True

def run_inference_and_explain():
    ordered_vals = [st.session_state.answers[q["key"]] for q in QUESTIONS]
    input_scaled = scaler.transform([ordered_vals])
    proba = float(model.predict_proba(input_scaled)[0][1])
    prediction_pct = proba * 100.0

    # Share result
    st.session_state.messages.append({
        "content": f"Your estimated risk (class 1 probability) is **{prediction_pct:.1f}%**.",
        "is_user": False
    })

    # SHAP explanation
    explainer = get_explainer(model)
    shap_vec = shap_positive_class_vector(explainer, model, input_scaled)

    # Build dataframe for plotting & table
    shap_df = pd.DataFrame({
        "Feature (code)": [q["key"] for q in QUESTIONS],
        "Feature (full name)": [FIELD_NAMES.get(q["key"], q["key"]) for q in QUESTIONS],
        "Value": ordered_vals,
        "SHAP": shap_vec
    })
    shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index)

    # Chart: top 10 contributors
    st.session_state.messages.append({
        "content": "Here are the top contributing factors (positive values push risk higher; negative lower):",
        "is_user": False
    })
    top_n = shap_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_n["Feature (full name)"][::-1], top_n["SHAP"][::-1])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_ylabel("Feature")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    st.pyplot(fig)

    # Table expander
    with st.expander("Show full SHAP table (codes + names)"):
        st.dataframe(
            shap_df[["Feature (code)", "Feature (full name)", "Value", "SHAP"]].reset_index(drop=True),
            use_container_width=True
        )

    # PDF download
    pdf_bytes = build_pdf_bytes(
        input_dict=st.session_state.answers,
        prediction_pct=prediction_pct,
        shap_vec=shap_vec,
        features_order=[q["key"] for q in QUESTIONS]
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="heart_risk_report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# Chat input box
# -----------------------------
if not st.session_state.completed:
    # Ask the current question (if not already shown this render)
    if st.session_state.q_index == 0 and len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "content": "Hi! I’ll ask a few quick questions to estimate your heart disease risk and explain the result.",
            "is_user": False
        })
        ask_current_question()
    elif st.session_state.q_index < len(QUESTIONS):
        ask_current_question()

    user_text = st.text_input("Type your answer and press Enter:", key="chat_input")
    if user_text:
        process_user_reply(user_text)
        # Clear input after processing
        st.experimental_rerun()
else:
    # Completed input collection
    if "explained" not in st.session_state:
        run_inference_and_explain()
        st.session_state.explained = True

    if st.button("Start over"):
        st.session_state.messages = []
        st.session_state.answers = {}
        st.session_state.q_index = 0
        st.session_state.completed = False
        if "explained" in st.session_state:
            del st.session_state["explained"]
        st.experimental_rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "For informational and triage purposes only — not medical advice. "
    "If you’re concerned about your heart health, please consult a clinician."
)

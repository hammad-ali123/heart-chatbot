import streamlit as st
from streamlit_chat import message
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Validation constraints & Qs
# -----------------------------
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
FEATURE_ORDER = [q["key"] for q in questions]
N_FEATURES = len(FEATURE_ORDER)

# -----------------------------
# Load model & scaler
# -----------------------------
model = joblib.load("model.pkl")     # LogisticRegression
scaler = joblib.load("scaler.pkl")   # Typically StandardScaler

# -----------------------------
# SHAP explainer (fix)
# -----------------------------
# We use a zero-centred background in *model input space*.
# If you used StandardScaler, a zero vector corresponds to average (mean) patient.
# This keeps explanations stable without needing training data here.
zero_background = np.zeros((1, N_FEATURES))

# Try the unified API first; fall back to LinearExplainer for older SHAP.
try:
    masker = shap.maskers.Independent(zero_background)
    explainer = shap.Explainer(model, masker)
except Exception:
    # Older SHAP versions: LinearExplainer(model, background)
    explainer = shap.LinearExplainer(model, zero_background)

def normalise_shap_values(shap_out):
    """
    Return a 1D numpy array for the first (and only) sample's SHAP values,
    regardless of SHAP version/backend.
    """
    # Newer SHAP: explainer(X) returns an Explanation object
    if hasattr(shap_out, "values") and hasattr(shap_out, "data"):
        # shap_out.values shape: (n_samples, n_features)
        return np.array(shap_out.values[0], dtype=float)

    # Older SHAP .shap_values(...) might return:
    # - np.ndarray of shape (n_samples, n_features)
    # - list of arrays (multiclass). For binary logistic regression,
    #   we typically get a single array.
    if isinstance(shap_out, np.ndarray):
        if shap_out.ndim == 2 and shap_out.shape[0] >= 1:
            return np.array(shap_out[0], dtype=float)
        raise ValueError("Unexpected SHAP ndarray shape.")
    if isinstance(shap_out, list) and len(shap_out) > 0:
        arr = shap_out[0]
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2 and arr.shape[0] >= 1:
                return np.array(arr[0], dtype=float)
    raise ValueError("Unrecognised SHAP output format.")

# -----------------------------
# PDF generator with SHAP table
# -----------------------------
def generate_pdf(input_data, prediction, shap_values_1d, features):
    field_names = {
        "age": "Age (years)",
        "sex": "Sex (0=Female, 1=Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (80–220 mm Hg)",
        "chol": "Cholesterol Level (100–700 mg/dl)",
        "fbs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
        "restecg": "Resting ECG Result (0–2)",
        "thalach": "Max Heart Rate Achieved (60–230)",
        "exang": "Exercise-Induced Angina (0=No, 1=Yes)",
        "oldpeak": "ST Depression by Exercise (0.0–7.0)",
        "slope": "Slope of Peak Exercise ST (0–2)",
        "ca": "Major Vessels Coloured (0–4)",
        "thal": "Thalassemia (0=Normal, 1=Fixed, 2=Reversible)"
    }

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Heart Disease Risk Assessment Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Input data
    elements.append(Paragraph("<b>Input Data:</b>", styles['Heading2']))
    for key, value in input_data.items():
        label = field_names.get(key, key)
        elements.append(Paragraph(f"{label}: {value}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Prediction
    elements.append(Paragraph(f"<b>Predicted Risk:</b> {round(prediction, 2)}%", styles['Heading2']))
    elements.append(Paragraph(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # SHAP breakdown table
    elements.append(Paragraph("<b>Feature Contributions (SHAP Analysis)</b>", styles['Heading2']))
    shap_df = pd.DataFrame({
        "Code": features,
        "Feature": [field_names.get(f, f) for f in features],
        "Value": [input_data[f] for f in features],
        "SHAP": shap_values_1d
    }).sort_values("SHAP", key=abs, ascending=False)

    table_data = [["Code", "Feature", "Value", "SHAP Impact"]]
    for _, row in shap_df.iterrows():
        shap_val = float(row["SHAP"])
        table_data.append([
            row["Code"],
            row["Feature"],
            row["Value"],
            Paragraph(f'<font color="{ "red" if shap_val > 0 else "blue" }">{shap_val:.3f}</font>', styles['Normal'])
        ])

    table = Table(table_data, hAlign='LEFT', colWidths=[50, 250, 50, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -----------------------------
# Helpers
# -----------------------------
def coerce_and_validate(key: str, raw_text: str):
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

# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# -----------------------------
# UI
# -----------------------------
st.title("💬 Heart Disease Risk Chatbot")

for i, msg in enumerate(st.session_state.chat_history):
    message(msg["text"], is_user=msg["is_user"], key=f"{'user' if msg['is_user'] else 'bot'}-{i}")

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

else:
    # Prepare model input in the correct feature order
    inputs = [st.session_state.answers[k] for k in FEATURE_ORDER]
    input_scaled = scaler.transform([inputs])
    prediction = model.predict_proba(input_scaled)[0][1] * 100

    # Risk message
    st.success(f"🧠 Your predicted heart disease risk is **{round(prediction, 2)}%**.")
    if prediction > 70:
        st.warning("⚠️ High risk! Please consult a doctor.")
    elif prediction > 40:
        st.info("🔍 Moderate risk. A check-up is recommended.")
    else:
        st.info("✅ Low risk. Great job keeping healthy!")

    # Pie chart
    st.markdown("### 📊 Risk Distribution")
    fig1, _ax1 = plt.subplots()
    _ax1.pie([prediction, 100 - prediction], labels=["At Risk", "No Risk"],
             colors=["red", "green"], autopct="%1.1f%%", startangle=90)
    _ax1.axis("equal")
    st.pyplot(fig1)

    # SHAP explainability (fixed)
    st.markdown("### 🧠 Feature Contributions (Explainability)")
    # Try unified API call first; if it fails, fall back to .shap_values(...)
    try:
        shap_raw = explainer(input_scaled)
    except Exception:
        shap_raw = explainer.shap_values(input_scaled)
    shap_1d = normalise_shap_values(shap_raw)

    shap_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "value": [st.session_state.answers[k] for k in FEATURE_ORDER],
        "shap": shap_1d
    }).sort_values("shap", key=abs, ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(shap_df["feature"], shap_df["shap"],
             color=["red" if x > 0 else "blue" for x in shap_df["shap"]])
    ax2.set_xlabel("SHAP Value (Impact on Prediction)")
    ax2.set_title("Top Feature Influences on Risk")
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # PDF download with SHAP table
    pdf = generate_pdf(st.session_state.answers, prediction, shap_1d, FEATURE_ORDER)
    st.download_button("📄 Download PDF Report", data=pdf,
                       file_name="heart_risk_report.pdf", mime="application/pdf")

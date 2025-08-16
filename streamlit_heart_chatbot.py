import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Heart Risk Chatbot (Explainable)", page_icon="❤️", layout="centered")

# -----------------------------
# Load model & scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_path = "model.pkl"      # matches your repo
    scaler_path = "scaler.pkl"    # matches your repo
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
# Feature meta
# -----------------------------
# Short codes (dataset columns) -> Full names for PDF and tables
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

# Hard feasibility constraints to prevent abnormal values
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

# Input questions in dataset column order
QUESTIONS = [
    {"key": "age", "text": "Enter your age:", "type": int},
    {"key": "sex", "text": "Sex (0 = Female, 1 = Male):", "type": int},
    {"key": "cp", "text": "Chest pain type (0–3 as per dataset encoding):", "type": int},
    {"key": "trestbps", "text": "Resting blood pressure (mm Hg):", "type": int},
    {"key": "chol", "text": "Serum cholesterol (mg/dl):", "type": int},
    {"key": "fbs", "text": "Fasting blood sugar > 120 mg/dl (0 = No, 1 = Yes):", "type": int},
    {"key": "restecg", "text": "Resting electrocardiographic results (0–2):", "type": int},
    {"key": "thalach", "text": "Maximum heart rate achieved:", "type": int},
    {"key": "exang", "text": "Exercise induced angina (0 = No, 1 = Yes):", "type": int},
    {"key": "oldpeak", "text": "ST depression induced by exercise relative to rest:", "type": float},
    {"key": "slope", "text": "Slope of peak exercise ST segment (0–2):", "type": int},
    {"key": "ca", "text": "Number of major vessels colored by fluoroscopy (0–3):", "type": int},
    {"key": "thal", "text": "Thalassemia (numeric code 0–3 as per dataset encoding):", "type": int},
]

# -----------------------------
# Utility: validate and cast
# -----------------------------
def validate_and_cast(key: str, raw_val: str):
    meta = CONSTRAINTS[key]
    typ = meta["type"]
    # cast
    try:
        val = typ(raw_val)
    except Exception:
        raise ValueError(f"'{FIELD_NAMES.get(key, key)}' must be of type {typ.__name__}.")
    # check choices or bounds
    if "choices" in meta:
        if val not in meta["choices"]:
            raise ValueError(
                f"'{FIELD_NAMES.get(key, key)}' must be one of {meta['choices']}."
            )
    else:
        lo, hi = meta["min"], meta["max"]
        if not (lo <= val <= hi):
            raise ValueError(
                f"'{FIELD_NAMES.get(key, key)}' must be between {lo} and {hi}."
            )
    return val

# -----------------------------
# SHAP explanation helper
# -----------------------------
@st.cache_resource
def get_explainer(trained_model):
    # Prefer TreeExplainer for tree models; fall back to generic
    try:
        return shap.TreeExplainer(trained_model)
    except Exception:
        return shap.Explainer(trained_model)

def compute_shap_for_positive_class(explainer, model, input_row_2d: np.ndarray) -> np.ndarray:
    """
    Returns a 1-D vector of SHAP contributions for the positive class (label==1).
    Handles both array and list-of-arrays outputs from SHAP across versions.
    """
    raw = explainer.shap_values(input_row_2d)
    if isinstance(raw, list):
        # binary tree models often return [neg_class, pos_class]
        if hasattr(model, "classes_"):
            pos_idx = list(model.classes_).index(1)
        else:
            pos_idx = 1
        shap_vec = raw[pos_idx][0]
    else:
        # raw is already (1, n_features) or (n_features,)
        shap_vec = raw[0] if raw.ndim > 1 else raw
    return np.asarray(shap_vec).ravel()

# -----------------------------
# PDF builder
# -----------------------------
def build_pdf_bytes(input_dict: dict, prediction_pct: float, shap_vec: np.ndarray, features_order: list):
    """
    Creates a PDF bytes object including prediction, inputs, and a SHAP contributions table
    showing short codes & full names.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("<b>Heart Risk Prediction Report</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 6))

    pred_text = Paragraph(f"Predicted risk (class 1 probability): <b>{prediction_pct:.1f}%</b>", styles["Normal"])
    story.append(pred_text)
    story.append(Spacer(1, 6))

    # Inputs table
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

    # SHAP contributions table
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
# UI
# -----------------------------
st.title("❤️ Explainable Heart Risk Chatbot")
st.write(
    "This tool estimates cardiovascular disease risk using a machine learning model and explains the factors that most influenced the prediction. "
    "**It does not provide a medical diagnosis.** For concerns, please consult a healthcare professional."
)

with st.expander("What are valid input ranges?"):
    lines = []
    for k, meta in CONSTRAINTS.items():
        if "choices" in meta:
            lines.append(f"- **{k}**: choices {meta['choices']}")
        else:
            lines.append(f"- **{k}**: {meta['min']} to {meta['max']}")
    st.markdown("\n".join(lines))

st.subheader("Enter your details")
inputs = {}
cols = st.columns(2)

for idx, q in enumerate(QUESTIONS):
    label = q["text"]
    k = q["key"]
    meta = CONSTRAINTS[k]
    with cols[idx % 2]:
        if "choices" in meta:
            # selectbox for discrete choices
            val = st.selectbox(label, meta["choices"], key=f"sb_{k}")
        else:
            if q["type"] is int:
                val = st.number_input(label, value=meta["min"], step=1, min_value=meta["min"], max_value=meta["max"], key=f"nb_{k}")
            else:
                val = st.number_input(label, value=float(meta["min"]), step=0.1, min_value=float(meta["min"]), max_value=float(meta["max"]), key=f"nb_{k}")
        inputs[k] = val

# -----------------------------
# Predict & Explain
# -----------------------------
st.markdown("---")
if st.button("Predict Risk"):
    try:
        # Validate & build ordered input list
        ordered_vals = []
        for q in QUESTIONS:
            k = q["key"]
            ordered_vals.append(validate_and_cast(k, inputs[k]))

        # Transform and predict
        input_scaled = scaler.transform([ordered_vals])
        proba = float(model.predict_proba(input_scaled)[0][1])
        prediction_pct = proba * 100.0

        st.subheader("Prediction")
        st.write(f"**Estimated risk (class 1 probability): {prediction_pct:.1f}%**")

        # SHAP explanation for positive class
        explainer = get_explainer(model)
        shap_vec = compute_shap_for_positive_class(explainer, model, input_scaled)

        # Local explanation bar chart (top 10 by absolute impact)
        shap_df = pd.DataFrame({
            "Feature (code)": [q["key"] for q in QUESTIONS],
            "Feature (full name)": [FIELD_NAMES.get(q["key"], q["key"]) for q in QUESTIONS],
            "Value": ordered_vals,
            "SHAP": shap_vec
        })
        shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index)

        st.subheader("Why this result? (Local explanation)")
        st.caption("Top contributing features to this individual prediction (SHAP). Positive values push towards higher risk; negative values towards lower risk.")
        top_n = shap_df.head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_n["Feature (full name)"][::-1], top_n["SHAP"][::-1])
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_ylabel("Feature")
        ax.axvline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)

        # Tabular view for transparency
        with st.expander("Show full SHAP table (codes + names)"):
            st.dataframe(
                shap_df[["Feature (code)", "Feature (full name)", "Value", "SHAP"]].reset_index(drop=True),
                use_container_width=True
            )

        # PDF download
        pdf_bytes = build_pdf_bytes(
            input_dict={k: inputs[k] for k in inputs},
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

        st.success("Done. You can save the PDF or adjust inputs to see how the explanation changes.")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "This tool is for informational and triage purposes only and does not constitute medical advice. "
    "If you are concerned about your heart health, please seek professional medical guidance."
)

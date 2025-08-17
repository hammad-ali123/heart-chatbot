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

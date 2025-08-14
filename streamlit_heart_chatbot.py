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

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# PDF generator with SHAP table
def generate_pdf(input_data, prediction, shap_values, features):
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

    # Predicti

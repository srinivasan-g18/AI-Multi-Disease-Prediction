import os
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------------------
# LOAD MODELS
# ---------------------------
MODEL_FILES = {
    "Kidney": "kidney_model.pkl",
    "Hypertension": "hypertension_model.pkl",
    "Stroke": "stroke_model.pkl",
    "Cancer": "cancer_model.pkl",
    "Liver": "liver_model.pkl",
    "Diabetes": "diabetes_model.pkl",
    "Heart": "heart_model.pkl",
}

models = {}
for name, filename in MODEL_FILES.items():
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            models[name] = pickle.load(f)

# ---------------------------
# APP CONFIG
# ---------------------------
st.set_page_config(page_title="Srini Health AI", page_icon="🩺", layout="centered")

# ---------------------------
# NAVIGATION MENU
# ---------------------------
menu = ["🏠 Home", "🩺 Disease Prediction"]
choice = st.sidebar.radio("Navigation", menu)

# ---------------------------
# HOMEPAGE
# ---------------------------
if choice == "🏠 Home":
    st.title("🩺 Welcome to Srini's Health AI")
    st.markdown("""
    ### 🌟 Your Personal AI Health Assistant    

    This tool helps you **predict disease risk** using Machine Learning.    
    Currently supports **7 diseases**:    
    - Kidney Disease    
    - Hypertension    
    - Stroke    
    - Cancer    
    - Liver Disease    
    - Diabetes    
    - Heart Disease    

    ---
    ### ⚡ How It Works:
    1. Select a disease from the sidebar    
    2. Enter patient details using simple sliders/dropdowns    
    3. Get a **risk prediction** (Low, Moderate, High)    
    4. Receive **health tips & recommendations**    
    5. Download your **Health Report as PDF**    

    > ⚠️ Disclaimer: This is an AI assistant, **not a doctor**.    
    Always consult a healthcare professional for medical advice.    
    """)
    st.success("👉 Use the left sidebar to start predicting!")

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif choice == "🩺 Disease Prediction":
    if not models:
        st.error("❌ No trained models found! Please add at least one.")
        st.stop()

    st.title("🩺 Disease Prediction")
    disease = st.selectbox("👉 Select a Disease", list(models.keys()))

    if disease not in models:
        st.error(f"❌ Model for {disease} not found.")
        st.stop()

    model = models[disease]
    patient_data = {}

    st.subheader(f"📋 Enter Patient Details for {disease}")

    # ---------------------------
    # INPUT FIELDS
    # ---------------------------
    if disease == "Kidney":
        patient_data = {
            "age": st.slider("Age", 1, 100, 48),
            "bp": st.slider("Blood Pressure", 50, 200, 80),
            "sg": st.slider("Specific Gravity", 1.0, 1.05, 1.02),
            "al": st.slider("Albumin", 0, 5, 1),
            "su": st.slider("Sugar", 0, 5, 0),
            "rbc": st.selectbox("RBC (0=normal,1=abnormal)", [0, 1]),
            "pc": st.selectbox("Pus Cell (0=normal,1=abnormal)", [0, 1]),
            "pcc": st.selectbox("Pus Cell Clumps", [0, 1]),
            "ba": st.selectbox("Bacteria", [0, 1]),
            "bgr": st.slider("Blood Glucose Random", 50, 300, 121),
            "bu": st.slider("Blood Urea", 10, 150, 36),
            "sc": st.slider("Serum Creatinine", 0.1, 10.0, 1.2),
            "sod": st.slider("Sodium", 100, 200, 137),
            "pot": st.slider("Potassium", 2.0, 10.0, 4.2),
            "hemo": st.slider("Hemoglobin", 5.0, 20.0, 15.4),
            "pcv": st.slider("PCV", 20, 60, 44),
            "wc": st.slider("WBC count", 2000, 20000, 7800),
            "rc": st.slider("RBC count", 2.0, 8.0, 5.2),
            "htn": st.selectbox("Hypertension", [0, 1]),
            "dm": st.selectbox("Diabetes Mellitus", [0, 1]),
            "cad": st.selectbox("Coronary Artery Disease", [0, 1]),
            "appet": st.selectbox("Appetite (0=poor,1=good)", [0, 1]),
            "pe": st.selectbox("Pedal Edema", [0, 1]),
            "ane": st.selectbox("Anemia", [0, 1]),
        }

    elif disease == "Hypertension":
        patient_data = {
            "age": st.slider("Age", 1, 100, 50),
            "education": st.slider("Education Level", 0, 5, 1),
            "currentSmoker": st.selectbox("Current Smoker", [0, 1]),
            "cigsPerDay": st.slider("Cigarettes Per Day", 0, 50, 5),
            "BPMeds": st.selectbox("On BP Medication", [0, 1]),
            "prevalentStroke": st.selectbox("Prevalent Stroke", [0, 1]),
            "prevalentHyp": st.selectbox("Prevalent Hypertension", [0, 1]),
            "diabetes": st.selectbox("Diabetes", [0, 1]),
            "totChol": st.slider("Total Cholesterol", 100, 400, 233),
            "sysBP": st.slider("Systolic BP", 90, 200, 138),
            "diaBP": st.slider("Diastolic BP", 60, 120, 80),
            "BMI": st.slider("BMI", 10.0, 50.0, 27.0),
            "heartRate": st.slider("Heart Rate", 40, 150, 77),
            "glucose": st.slider("Glucose", 50, 250, 80),
        }

    elif disease == "Stroke":
        patient_data = {
            "gender": st.selectbox("Gender (0=Female,1=Male)", [0, 1]),
            "age": st.slider("Age", 1, 100, 45),
            "hypertension": st.selectbox("Hypertension", [0, 1]),
            "heart_disease": st.selectbox("Heart Disease", [0, 1]),
            "ever_married": st.selectbox("Ever Married", [0, 1]),
            "work_type": st.slider("Work Type (0-4)", 0, 4, 2),
            "Residence_type": st.selectbox("Residence Type (0=Rural,1=Urban)", [0, 1]),
            "avg_glucose_level": st.slider("Avg Glucose Level", 50, 300, 90),
            "bmi": st.slider("BMI", 10.0, 50.0, 28.0),
            "smoking_status": st.slider("Smoking Status (0-3)", 0, 3, 1),
        }

    elif disease == "Cancer":
        patient_data = {
            "radius_mean": st.slider("Radius Mean", 5.0, 30.0, 14.2),
            "texture_mean": st.slider("Texture Mean", 5.0, 40.0, 20.1),
            "perimeter_mean": st.slider("Perimeter Mean", 20.0, 200.0, 90.2),
            "area_mean": st.slider("Area Mean", 100.0, 2500.0, 600.1),
            "smoothness_mean": st.slider("Smoothness Mean", 0.05, 0.2, 0.1),
            "compactness_mean": st.slider("Compactness Mean", 0.0, 1.0, 0.2),
            "concavity_mean": st.slider("Concavity Mean", 0.0, 1.0, 0.3),
            "concave points_mean": st.slider("Concave Points Mean", 0.0, 1.0, 0.1),
            "symmetry_mean": st.slider("Symmetry Mean", 0.0, 1.0, 0.2),
            "fractal_dimension_mean": st.slider("Fractal Dimension Mean", 0.0, 0.2, 0.06),
        }

    elif disease == "Liver":
        patient_data = {
            "Age": st.slider("Age", 1, 100, 45),
            "Gender": st.selectbox("Gender (0=Female,1=Male)", [0, 1]),
            "Total_Bilirubin": st.slider("Total Bilirubin", 0.0, 10.0, 0.9),
            "Direct_Bilirubin": st.slider("Direct Bilirubin", 0.0, 5.0, 0.3),
            "Alkaline_Phosphotase": st.slider("Alkaline Phosphotase", 50, 1000, 200),
            "Alamine_Aminotransferase": st.slider("Alamine Aminotransferase", 0, 200, 30),
            "Aspartate_Aminotransferase": st.slider("Aspartate Aminotransferase", 0, 200, 40),
            "Total_Protiens": st.slider("Total Proteins", 2.0, 9.0, 6.8),
            "Albumin": st.slider("Albumin", 1.0, 6.0, 3.5),
            "Albumin_and_Globulin_Ratio": st.slider("Albumin & Globulin Ratio", 0.0, 3.0, 1.1),
        }

    elif disease == "Diabetes":
        patient_data = {
            "Pregnancies": st.slider("Pregnancies", 0, 15, 2),
            "Glucose": st.slider("Glucose", 50, 250, 120),
            "BloodPressure": st.slider("Blood Pressure", 40, 150, 70),
            "SkinThickness": st.slider("Skin Thickness", 0, 100, 20),
            "Insulin": st.slider("Insulin", 0, 900, 85),
            "BMI": st.slider("BMI", 10.0, 60.0, 28.5),
            "DiabetesPedigreeFunction": st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5),
            "Age": st.slider("Age", 1, 100, 35),
        }

    elif disease == "Heart":
        patient_data = {
            "age": st.slider("Age", 1, 100, 52),
            "sex": st.selectbox("Sex (0=Female,1=Male)", [0, 1]),
            "cp": st.slider("Chest Pain Type (0-3)", 0, 3, 0),
            "trestbps": st.slider("Resting BP", 80, 200, 130),
            "chol": st.slider("Cholesterol", 100, 600, 240),
            "fbs": st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
            "restecg": st.slider("Resting ECG (0-2)", 0, 2, 1),
            "thalach": st.slider("Max Heart Rate Achieved", 70, 220, 150),
            "exang": st.selectbox("Exercise Induced Angina", [0, 1]),
            "oldpeak": st.slider("Oldpeak", 0.0, 10.0, 1.0),
            "slope": st.slider("Slope (0-2)", 0, 2, 2),
            "ca": st.slider("Major Vessels (0-3)", 0, 3, 0),
            "thal": st.slider("Thal (0-3)", 0, 3, 2),
        }

    # ---------------------------
    # PREDICTION
    # ---------------------------
    if st.button("🔮 Predict"):
        df = pd.DataFrame([patient_data])

        # Align columns to model
        if hasattr(model, "feature_names_in_"):
            missing_cols = set(model.feature_names_in_) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[model.feature_names_in_]

        prob = model.predict_proba(df)[0][1] * 100

        # Risk categorization
        if prob < 33:
            risk = "Low"
        elif prob < 66:
            risk = "Moderate"
        else:
            risk = "High"

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': f"{disease} Risk (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if risk == "High" else "orange" if risk == "Moderate" else "green"},
                   'steps': [
                       {'range': [0, 33], 'color': "lightgreen"},
                       {'range': [33, 66], 'color': "yellow"},
                       {'range': [66, 100], 'color': "pink"}
                   ]}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 🧾 Result: **{risk} Risk**")
        if risk == "High":
            recommendation = "⚠️ Consult a doctor immediately."
            st.error(recommendation)
        elif risk == "Moderate":
            recommendation = "⚠️ Monitor closely & improve lifestyle."
            st.warning(recommendation)
        else:
            recommendation = "✅ Maintain healthy lifestyle."
            st.success(recommendation)

        # ---------------------------
        # PDF REPORT
        # ---------------------------
        report_filename = f"{disease}_report.pdf"
        c = canvas.Canvas(report_filename, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(200, 800, "Health Prediction Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 770, f"Disease: {disease}")
        c.drawString(50, 750, f"Risk Probability: {prob:.2f}%")
        c.drawString(50, 730, f"Risk Level: {risk}")
        c.drawString(50, 710, f"Recommendation: {recommendation}")

        c.drawString(50, 680, "Patient Details:")
        y = 660
        for key, value in patient_data.items():
            c.drawString(60, y, f"{key}: {value}")
            y -= 20
            if y < 100:
                c.showPage()
                y = 780
        c.save()

        with open(report_filename, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f,
                file_name=report_filename,
                mime="application/pdf"
            )

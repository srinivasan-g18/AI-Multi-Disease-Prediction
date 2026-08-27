import pickle

# ---------------------------
# HOMEPAGE / WELCOME
# ---------------------------
def homepage():
    print("="*60)
    print(" 🩺 WELCOME TO HEALTH AI PREDICTION SYSTEM 🩺 ")
    print("="*60)
    print("This app predicts risk levels for:")
    print("1. Heart Disease")
    print("2. Diabetes")
    print("3. Chronic Kidney Disease")
    print("4. Hypertension (10-Year CHD Risk)")
    print("5. Cancer (Breast Cancer)")
    print("6. Liver Disease")
    print("7. Stroke")
    print("-"*60)
    print("👉 Enter patient details, and the app will give:")
    print("   - Probability (%)")
    print("   - Risk Level (Low / Moderate / High)")
    print("   - Health Recommendations")
    print("="*60)


# ---------------------------
# RISK FUNCTION
# ---------------------------
def get_risk(prob):
    if prob < 40:
        return "Low"
    elif prob < 70:
        return "Moderate"
    else:
        return "High"


# ---------------------------
# RECOMMENDATIONS
# ---------------------------
recommendations = {
    "Heart": {
        "Low": "✅ Maintain healthy lifestyle and regular check-ups.",
        "Moderate": "⚠️ Improve diet, exercise, and monitor blood pressure.",
        "High": "❌ Consult cardiologist immediately, strict lifestyle changes."
    },
    "Diabetes": {
        "Low": "✅ Balanced diet, regular exercise.",
        "Moderate": "⚠️ Monitor blood sugar and reduce sugar intake.",
        "High": "❌ Consult doctor, strict diet control, possible medication."
    },
    "Kidney": {
        "Low": "✅ Stay hydrated, avoid excessive salt.",
        "Moderate": "⚠️ Monitor kidney function regularly.",
        "High": "❌ Consult nephrologist urgently, strict care needed."
    },
    "Hypertension": {
        "Low": "✅ Maintain active lifestyle, reduce salt intake.",
        "Moderate": "⚠️ Monitor blood pressure regularly.",
        "High": "❌ Consult doctor, possible medication required."
    },
    "Cancer": {
        "Low": "✅ Maintain regular screenings and healthy lifestyle.",
        "Moderate": "⚠️ More tests may be needed.",
        "High": "❌ Immediate consultation with oncologist recommended."
    },
    "Liver": {
        "Low": "✅ Avoid alcohol, maintain healthy weight.",
        "Moderate": "⚠️ Regular liver function tests required.",
        "High": "❌ Consult hepatologist immediately."
    },
    "Stroke": {
        "Low": "✅ Control blood pressure, avoid smoking.",
        "Moderate": "⚠️ Manage cholesterol and diabetes risks.",
        "High": "❌ High stroke risk! Consult neurologist urgently."
    }
}


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_disease(model_file, features, disease_name):
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    prob = model.predict_proba([features])[0][1] * 100
    risk = get_risk(prob)

    print(f"\n🔍 {disease_name} Prediction Result:")
    print(f"   Probability: {prob:.2f}%")
    print(f"   Risk Level: {risk}")
    print(f"   Recommendation: {recommendations[disease_name][risk]}")
    print("-"*60)


# ---------------------------
# MAIN APP
# ---------------------------
if __name__ == "__main__":
    homepage()
    choice = input("👉 Choose disease (1-7): ")

    if choice == "1":  # Heart
        print("\nEnter details for Heart Disease:")
        age = int(input("Age: "))
        sex = int(input("Sex (1=Male, 0=Female): "))
        cp = int(input("Chest Pain Type (0-3): "))
        trestbps = int(input("Resting Blood Pressure: "))
        chol = int(input("Cholesterol: "))
        fbs = int(input("Fasting Blood Sugar > 120 (1/0): "))
        restecg = int(input("Resting ECG (0-2): "))
        thalach = int(input("Max Heart Rate: "))
        exang = int(input("Exercise Induced Angina (1/0): "))
        oldpeak = float(input("ST Depression: "))
        slope = int(input("Slope (0-2): "))
        ca = int(input("Number of Major Vessels (0-3): "))
        thal = int(input("Thal (0=Normal,1=Fixed,2=Reversible): "))
        features = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]
        predict_disease("heart_model.pkl", features, "Heart")

    elif choice == "2":  # Diabetes
        print("\nEnter details for Diabetes:")
        preg = int(input("Pregnancies: "))
        glucose = int(input("Glucose: "))
        bp = int(input("Blood Pressure: "))
        skin = int(input("Skin Thickness: "))
        insulin = int(input("Insulin: "))
        bmi = float(input("BMI: "))
        dpf = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age: "))
        features = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        predict_disease("diabetes_model.pkl", features, "Diabetes")

    elif choice == "3":  # Kidney
        print("\nEnter details for Kidney Disease:")
        age = int(input("Age: "))
        bp = int(input("Blood Pressure: "))
        sg = float(input("Specific Gravity (e.g. 1.02): "))
        al = int(input("Albumin: "))
        su = int(input("Sugar: "))
        rbc = int(input("RBC (0=Normal, 1=Abnormal): "))
        pc = int(input("Pus Cell (0=Normal, 1=Abnormal): "))
        pcc = int(input("Pus Cell Clumps (0/1): "))
        ba = int(input("Bacteria (0/1): "))
        htn = int(input("Hypertension (0/1): "))
        dm = int(input("Diabetes Mellitus (0/1): "))
        features = [age, bp, sg, al, su, rbc, pc, pcc, ba, htn, dm]
        predict_disease("kidney_model.pkl", features, "Kidney")

    elif choice == "4":  # Hypertension
        print("\nEnter details for Hypertension (Framingham):")
        age = int(input("Age: "))
        sex = int(input("Sex (1=Male, 0=Female): "))
        currentSmoker = int(input("Current Smoker (1/0): "))
        cigsPerDay = float(input("Cigarettes Per Day: "))
        bpMed = int(input("On BP Meds (1/0): "))
        prevalentStroke = int(input("History of Stroke (1/0): "))
        prevalentHyp = int(input("Hypertension (1/0): "))
        diabetes = int(input("Diabetes (1/0): "))
        totChol = float(input("Total Cholesterol: "))
        sysBP = float(input("Systolic BP: "))
        diaBP = float(input("Diastolic BP: "))
        BMI = float(input("BMI: "))
        heartRate = float(input("Heart Rate: "))
        glucose = float(input("Glucose: "))
        features = [age, sex, currentSmoker, cigsPerDay, bpMed,
                    prevalentStroke, prevalentHyp, diabetes, totChol,
                    sysBP, diaBP, BMI, heartRate, glucose]
        predict_disease("hypertension_model.pkl", features, "Hypertension")

    elif choice == "5":  # Cancer
        print("\nEnter details for Cancer (Breast):")
        radius_mean = float(input("Radius Mean: "))
        texture_mean = float(input("Texture Mean: "))
        perimeter_mean = float(input("Perimeter Mean: "))
        area_mean = float(input("Area Mean: "))
        smoothness_mean = float(input("Smoothness Mean: "))
        compactness_mean = float(input("Compactness Mean: "))
        concavity_mean = float(input("Concavity Mean: "))
        concave_points_mean = float(input("Concave Points Mean: "))
        symmetry_mean = float(input("Symmetry Mean: "))
        fractal_dimension_mean = float(input("Fractal Dimension Mean: "))
        features = [radius_mean, texture_mean, perimeter_mean, area_mean,
                    smoothness_mean, compactness_mean, concavity_mean,
                    concave_points_mean, symmetry_mean, fractal_dimension_mean]
        predict_disease("cancer_model.pkl", features, "Cancer")

    elif choice == "6":  # Liver
        print("\nEnter details for Liver Disease:")
        age = int(input("Age: "))
        gender = int(input("Gender (1=Male, 0=Female): "))
        tb = float(input("Total Bilirubin: "))
        alkphos = int(input("Alkaline Phosphotase: "))
        sgpt = int(input("SGPT: "))
        sgot = int(input("SGOT: "))
        features = [age, gender, tb, alkphos, sgpt, sgot]
        predict_disease("liver_model.pkl", features, "Liver")

    elif choice == "7":  # Stroke
        print("\nEnter details for Stroke Risk:")
        age = int(input("Age: "))
        gender = int(input("Gender (1=Male, 0=Female): "))
        hypertension = int(input("Hypertension (1/0): "))
        heart_disease = int(input("Heart Disease (1/0): "))
        avg_glucose = float(input("Average Glucose Level: "))
        bmi = float(input("BMI: "))
        smoking_status = int(input("Smoking Status (0=Never,1=Former,2=Smokes): "))
        married = int(input("Ever Married (1/0): "))
        features = [age, gender, hypertension, heart_disease,
                    avg_glucose, bmi, smoking_status, married]
        predict_disease("stroke_model.pkl", features, "Stroke")

    else:
        print("❌ Invalid choice. Please run again.")

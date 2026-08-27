import pandas as pd
import pickle

# Load model
with open("hypertension_model.pkl", "rb") as f:
    model = pickle.load(f)

# Features (must match training order)
feature_cols = [
    "age", "male", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

# === Example patients ===
examples = {
    "Low Risk": {
        "age": 35, "male": 0, "currentSmoker": 0, "cigsPerDay": 0,
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0,
        "totChol": 180, "sysBP": 110, "diaBP": 70, "BMI": 22,
        "heartRate": 72, "glucose": 85
    },
    "Moderate Risk": {
        "age": 50, "male": 1, "currentSmoker": 1, "cigsPerDay": 10,
        "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0,
        "totChol": 220, "sysBP": 135, "diaBP": 85, "BMI": 27,
        "heartRate": 78, "glucose": 105
    },
    "High Risk": {
        "age": 65, "male": 1, "currentSmoker": 1, "cigsPerDay": 20,
        "BPMeds": 1, "prevalentStroke": 1, "prevalentHyp": 1, "diabetes": 1,
        "totChol": 260, "sysBP": 160, "diaBP": 100, "BMI": 32,
        "heartRate": 90, "glucose": 140
    }
}

# === Run predictions ===
for label, sample in examples.items():
    df = pd.DataFrame([sample], columns=feature_cols)
    prob = model.predict_proba(df)[0][1] * 100
    
    if prob < 30:
        risk = "Low"
    elif prob < 60:
        risk = "Moderate"
    else:
        risk = "High"
    
    print(f"\n🔹 {label} Example:")
    print(f"   Probability: {prob:.2f}%")
    print(f"   Risk Level: {risk}")
    print("   Recommendation:", 
          "Maintain healthy lifestyle ✅" if risk=="Low" else 
          "Regular checkups & lifestyle changes ⚠️" if risk=="Moderate" else 
          "Consult doctor immediately 🚨")

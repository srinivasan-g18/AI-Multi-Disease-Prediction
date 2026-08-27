import pandas as pd
import pickle

# Load trained model + metadata
with open("stroke_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_names = saved["features"]
encoders = saved["encoders"]

# Example patient data (raw, before encoding)
sample = {
    "gender": "Male",         # Male / Female / Other
    "age": 67,
    "hypertension": 1,        # 0 = No, 1 = Yes
    "heart_disease": 1,       # 0 = No, 1 = Yes
    "ever_married": "Yes",    # Yes / No
    "work_type": "Private",   # Private / Self-employed / Govt_job / children / Never_worked
    "Residence_type": "Urban",# Urban / Rural
    "avg_glucose_level": 250.5,
    "bmi": 30.5,
    "smoking_status": "formerly smoked"  # formerly smoked / never smoked / smokes / Unknown
}

# Convert to DataFrame
df = pd.DataFrame([sample])

# Encode categorical columns using saved encoders
for col, le in encoders.items():
    df[col] = le.transform(df[col].astype(str))

# Reorder columns to match training
df = df[feature_names]

# Predict probability
prob = model.predict_proba(df)[0][1] * 100
risk = "Low"
if prob >= 40 and prob < 70:
    risk = "Moderate"
elif prob >= 70:
    risk = "High"

print(f"🔍 Stroke Prediction Result")
print(f"   Probability: {prob:.2f}%")
print(f"   Risk Level: {risk}")

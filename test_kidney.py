import pandas as pd
import pickle

# Load trained model
with open("kidney_model.pkl", "rb") as f:   # note: models/ folder
    model = pickle.load(f)

# Example input (values chosen to simulate a patient)
sample = {
    "age": 48,
    "bp": 80,
    "sg": 1.02,
    "al": 1,
    "su": 0,
    "rbc": 1,   # encoded (normal/abnormal after LabelEncoder)
    "pc": 0,
    "pcc": 0,
    "ba": 0,
    "htn": 1,
    "dm": 0
}

# Convert to DataFrame (must match training feature order!)
feature_order = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm"]
df = pd.DataFrame([[sample[feat] for feat in feature_order]], columns=feature_order)

# Predict probability
prob = model.predict_proba(df)[0][1] * 100

# Risk level (same thresholds you use in main app)
if prob < 40:
    risk = "Low"
elif prob < 70:
    risk = "Moderate"
else:
    risk = "High"

print(f"Probability: {prob:.2f}%")
print(f"Risk Level: {risk}")

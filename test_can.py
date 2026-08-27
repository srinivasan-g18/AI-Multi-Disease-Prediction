import pandas as pd
import pickle

# Load trained model + feature names
with open("cancer_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_names = saved["features"]

# Load dataset again to calculate averages (for missing features)
data = pd.read_csv("data.csv")
if "id" in data.columns:
    data = data.drop("id", axis=1)
if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
feature_means = data.drop("diagnosis", axis=1).mean()

# Example input (only some features given, rest will be filled with mean)
sample = {
    "radius_mean": 20.5,
    "texture_mean": 17.8,
    "perimeter_mean": 130.0,
    "area_mean": 1200.0,
    "smoothness_mean": 0.10,
    "compactness_mean": 0.20,
    "concavity_mean": 0.25,
    "concave points_mean": 0.12,
    "symmetry_mean": 0.20,
    "fractal_dimension_mean": 0.06
}

# Fill missing features with mean values
full_sample = {}
for col in feature_names:
    if col in sample:
        full_sample[col] = sample[col]
    else:
        full_sample[col] = feature_means[col]

# Convert to DataFrame
df = pd.DataFrame([full_sample])
df = df[feature_names]  # Ensure same order

# Predict probability
prob = model.predict_proba(df)[0][1] * 100
risk = "Low"
if prob >= 40 and prob < 70:
    risk = "Moderate"
elif prob >= 70:
    risk = "High"

print("🔍 Breast Cancer Prediction")
print(f"   Probability: {prob:.2f}%")
print(f"   Risk Level: {risk}")

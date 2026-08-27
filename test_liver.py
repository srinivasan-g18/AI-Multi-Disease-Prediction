import pandas as pd
import pickle

# 1. Load trained model + feature names
with open("liver_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
features = saved["features"]

# 2. Load dataset again to get averages for missing values
data = pd.read_csv("indian_liver_patient.csv")

# Encode Gender same way as training
data["Gender"] = data["Gender"].map({"Female": 0, "Male": 1})

# Target fix
data["Dataset"] = data["Dataset"].map({1: 1, 2: 0})

# Fill missing with median
data = data.fillna(data.median(numeric_only=True))

# Compute median values for fallback
median_values = data[features].median()

# 3. Example patient input (you can change values here)
sample = {
    "Age": 55,
    "Gender": 1,          # 1 = Male, 0 = Female
    "Total_Bilirubin": 2.5,
    "Direct_Bilirubin": 1.2,
    "Alkaline_Phosphotase": 210,
    "Alamine_Aminotransferase": 45,
    "Aspartate_Aminotransferase": 35,
    "Total_Protiens": 6.5,
    "Albumin": 3.2,
    "Albumin_and_Globulin_Ratio": 0.9
}

# 4. Fill missing fields with dataset median
for col in features:
    if col not in sample:
        sample[col] = median_values[col]

# 5. Convert to DataFrame (keep correct column order)
df = pd.DataFrame([sample])[features]

# 6. Predict probability
prob = model.predict_proba(df)[0][1] * 100

# 7. Categorize risk
if prob < 33:
    risk = "Low"
elif prob < 66:
    risk = "Moderate"
else:
    risk = "High"

# 8. Output
print(f"🔍 Probability of Liver Disease: {prob:.2f}%")
print(f"⚠️ Risk Level: {risk}")
print("Recommendation:", 
      "Consult a liver specialist immediately." if risk == "High" 
      else "Schedule a checkup soon." if risk == "Moderate" 
      else "Maintain healthy lifestyle.")

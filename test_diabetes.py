import pickle
import pandas as pd

# 1. Load saved model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Example patient data (must match dataset order)
cols = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

sample = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=cols)

# 3. Predict probability
prob = model.predict_proba(sample)[0][1] * 100

# 4. Risk level
if prob < 40:
    risk = "Low"
elif prob < 70:
    risk = "Moderate"
else:
    risk = "High"

# 5. Recommendations
recommendations = {
    "Low": "✅ Maintain balanced diet and regular exercise.",
    "Moderate": "⚠️ Monitor blood sugar regularly, reduce sugar intake.",
    "High": "❌ Consult doctor, strict diet control, and possible medication."
}

print(f"Probability: {prob:.2f}%")
print(f"Risk Level: {risk}")
print(f"Recommendation: {recommendations[risk]}")

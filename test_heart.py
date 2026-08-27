import pickle
import pandas as pd

# Load saved model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Column names (same as training)
cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal"
]

# Example input as DataFrame
sample = pd.DataFrame([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]], columns=cols)

# Predict probability
prob = model.predict_proba(sample)[0][1] * 100

# Risk level
if prob < 40:
    risk = "Low"
elif prob < 70:
    risk = "Moderate"
else:
    risk = "High"

# Recommendations
recommendations = {
    "Low": "✅ Maintain healthy lifestyle and regular check-ups.",
    "Moderate": "⚠️ Improve diet, exercise, and monitor blood pressure.",
    "High": "❌ Consult cardiologist immediately, strict lifestyle changes."
}

print(f"Probability: {prob:.2f}%")
print(f"Risk Level: {risk}")
print(f"Recommendation: {recommendations[risk]}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load dataset
data = pd.read_csv("framingham.csv")  # Make sure file is in same folder or adjust path

# 2. Target variable is already 0/1 -> "TenYearCHD"
target = "TenYearCHD"

# 3. Drop useless columns if any
useless_cols = [col for col in data.columns if "unnamed" in col.lower()]
data = data.drop(columns=useless_cols, errors="ignore")

# 4. Handle missing values
data = data.dropna()

# 5. Encode categorical features if needed
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# 6. Features (important ones from Framingham dataset)
feature_cols = [
    "age", "male", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

X = data[feature_cols]
y = data[target]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# 9. Train model
model.fit(X_train, y_train)

# 10. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 11. Save model
os.makedirs("models", exist_ok=True)
with open("hypertension_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as hypertension_model.pkl")

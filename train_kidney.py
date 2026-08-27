import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load dataset
data = pd.read_csv("kidney_disease.csv")

# 2. Replace target labels
data["classification"] = data["classification"].replace({"ckd": 1, "notckd": 0})

# 3. Drop useless columns (id, name, etc.)
useless_cols = [col for col in data.columns if col.lower() in ["id", "name"] or "unnamed" in col.lower()]
data = data.drop(columns=useless_cols, errors="ignore")

# 4. Encode categorical columns
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# 5. Handle missing values
for col in data.columns:
    if data[col].dtype in ["float64", "int64"]:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna(data[col].mode()[0])

# 6. Select ONLY the columns your app uses (in same order!)
feature_cols = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm"]

X = data[feature_cols]
y = data["classification"]

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

# 11. Save trained model
os.makedirs("models", exist_ok=True)
with open("kidney_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as kidney_model.pkl")
print("✅ Feature order:", feature_cols)

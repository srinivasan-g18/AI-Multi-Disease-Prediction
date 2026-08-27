import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Column names for the dataset
cols = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
]

# 1. Load dataset
data = pd.read_csv("heart.csv", names=cols)

# 2. Replace '?' with NaN
data = data.replace("?", pd.NA)

# 3. Drop rows with missing values
data = data.dropna()

# 4. Convert all columns to numeric
data = data.apply(pd.to_numeric)

# 5. Convert target into binary (0 = no disease, 1 = disease)
data["target"] = (data["target"] > 0).astype(int)

# 6. Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Create pipeline (scaler + logistic regression)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# 9. Train model
model.fit(X_train, y_train)

# 10. Evaluate model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 11. Save trained model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as heart_model.pkl")

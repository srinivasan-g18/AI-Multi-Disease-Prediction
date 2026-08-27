import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
data = pd.read_csv("diabetes.csv")

# 2. Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 7. Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as diabetes_model.pkl")

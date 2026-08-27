import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load dataset
data = pd.read_csv("indian_liver_patient.csv")

# 2. Encode categorical column "Gender"
if "Gender" in data.columns:
    le = LabelEncoder()
    data["Gender"] = le.fit_transform(data["Gender"].astype(str))

# 3. Fix target column (1 = disease, 2 = no disease → make 1/0)
data["Dataset"] = data["Dataset"].map({1: 1, 2: 0})

# 4. Handle missing values
data = data.fillna(data.median(numeric_only=True))

# 5. Features and target
X = data.drop("Dataset", axis=1)
y = data["Dataset"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Build pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])

# 8. Train
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save model along with feature names
os.makedirs("models", exist_ok=True)
with open("liver_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as liver_model.pkl ")

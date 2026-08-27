import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# ✅ Load Heart dataset with correct headers
def load_heart_dataset(file_path):
    heart_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    return pd.read_csv(file_path, header=None, names=heart_columns)

# ✅ Train Heart Model
def train_heart_model(file_path):
    print("\n===== HEART MODEL =====")

    try:
        data = load_heart_dataset(file_path)
        print("✅ Dataset loaded:", file_path)
        print("Columns:", list(data.columns))
        print("Target column unique values:", data["target"].unique()[:10])

        # Replace missing values
        data = data.replace("?", np.nan).dropna()

        X = data.drop("target", axis=1)
        y = data["target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Balance dataset with SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train model
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Train metrics
        print("\n--- Train Metrics ---")
        print("Accuracy   :", accuracy_score(y_train, y_pred_train))
        print("Precision  :", precision_score(y_train, y_pred_train, average="weighted", zero_division=0))
        print("Recall     :", recall_score(y_train, y_pred_train, average="weighted"))
        print("F1-score   :", f1_score(y_train, y_pred_train, average="weighted"))
        print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

        # Test metrics
        print("\n--- Test Metrics ---")
        print("Accuracy   :", accuracy_score(y_test, y_pred_test))
        print("Precision  :", precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
        print("Recall     :", recall_score(y_test, y_pred_test, average="weighted"))
        print("F1-score   :", f1_score(y_test, y_pred_test, average="weighted"))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

    except Exception as e:
        print("❌ ERROR in Heart model:", e)


# ================= Run Only Heart Model =================
train_heart_model("heart.csv")

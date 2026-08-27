import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# ================= Helper Function =================
def train_model(name, file_path, target_col, use_smote=False, drop_cols=None, scale=False):
    print(f"\n===== {name.upper()} MODEL =====")

    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Drop unwanted columns
        if drop_cols:
            data = data.drop(columns=drop_cols, errors="ignore")

        # Clean missing values
        data = data.replace("?", np.nan).dropna()

        # Encode categorical columns
        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))

        # Encode target if categorical
        if data[target_col].dtype == "object":
            data[target_col] = LabelEncoder().fit_transform(data[target_col].astype(str))

        # Check empty dataset
        if data.shape[0] == 0:
            print("❌ ERROR: Dataset is empty after cleaning.")
            return

        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Feature scaling
        if scale:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle imbalance
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Model
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

        # --- Train Metrics ---
        print("\n--- Train Metrics ---")
        print("Accuracy   :", accuracy_score(y_train, y_pred_train))
        print("Precision  :", precision_score(y_train, y_pred_train, average="weighted", zero_division=0))
        print("Recall     :", recall_score(y_train, y_pred_train, average="weighted"))
        print("F1-score   :", f1_score(y_train, y_pred_train, average="weighted"))
        print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

        # --- Test Metrics ---
        print("\n--- Test Metrics ---")
        print("Accuracy   :", accuracy_score(y_test, y_pred_test))
        print("Precision  :", precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
        print("Recall     :", recall_score(y_test, y_pred_test, average="weighted"))
        print("F1-score   :", f1_score(y_test, y_pred_test, average="weighted"))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

        # --- ROC Curve (only for binary targets) ---
        if y_proba_test is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba_test)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend(loc="lower right")
            plt.savefig(f"{name}_roc_curve.png")
            plt.close()
            print(f"📊 ROC curve saved as {name}_roc_curve.png")

    except Exception as e:
        print(f"❌ ERROR in {name} model:", e)


# ================= Run All Models =================
train_model("Kidney", "kidney_disease.csv", target_col="classification", use_smote=True, drop_cols=["id"])
train_model("Heart", "heart.csv", target_col="target", use_smote=True, scale=True)
train_model("Diabetes", "diabetes.csv", target_col="Outcome", use_smote=True, scale=True)
train_model("Liver", "indian_liver_patient.csv", target_col="Dataset", use_smote=True)
train_model("Cancer", "data.csv", target_col="diagnosis", use_smote=False, drop_cols=["id", "Unnamed: 32"], scale=True)
train_model("Stroke", "healthcare-dataset-stroke-data.csv", target_col="stroke", use_smote=True, drop_cols=["id"])
train_model("Hypertension", "framingham.csv", target_col="TenYearCHD", use_smote=True)

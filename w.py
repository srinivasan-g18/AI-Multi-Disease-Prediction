import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def train_model(name, df, target_col, use_smote=False):
    print(f"\n===== {name.upper()} MODEL =====")
    try:
        print(f"✅ Dataset loaded: {name}.csv")
        print("Columns:", list(df.columns))

        # Drop useless ID-like columns
        df = df.loc[:, ~df.columns.str.contains('id', case=False)]
        df = df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]

        # Encode categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()  # remove stray tabs/spaces
                df[col] = LabelEncoder().fit_transform(df[col])

        # Handle missing values
        df = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(df), columns=df.columns)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        print("Target column unique values:", np.unique(y))

        if len(X) == 0:
            raise ValueError("❌ ERROR: Dataset is empty after cleaning.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Train metrics
        y_train_pred = model.predict(X_train)
        print("\n--- Train Metrics ---")
        print("Accuracy   :", accuracy_score(y_train, y_train_pred))
        print("Precision  :", precision_score(y_train, y_train_pred, average='weighted'))
        print("Recall     :", recall_score(y_train, y_train_pred, average='weighted'))
        print("F1-score   :", f1_score(y_train, y_train_pred, average='weighted'))
        print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

        # Test metrics
        y_test_pred = model.predict(X_test)
        print("\n--- Test Metrics ---")
        print("Accuracy   :", accuracy_score(y_test, y_test_pred))
        print("Precision  :", precision_score(y_test, y_test_pred, average='weighted'))
        print("Recall     :", recall_score(y_test, y_test_pred, average='weighted'))
        print("F1-score   :", f1_score(y_test, y_test_pred, average='weighted'))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    except Exception as e:
        print(f"❌ ERROR in {name} model:", e)


# ==========================
# Load and Train Models
# ==========================

# Kidney
kidney = pd.read_csv("kidney_disease.csv")
train_model("Kidney", kidney, target_col="classification")

# Heart (fix column names!)
try:
    heart = pd.read_csv("heart.csv")
    if not "target" in heart.columns:
        # Replace with standard heart dataset headers
        heart.columns = [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"
        ]
    train_model("Heart", heart, target_col="target")
except Exception as e:
    print("❌ ERROR in Heart model:", e)

# Diabetes
diabetes = pd.read_csv("diabetes.csv")
train_model("Diabetes", diabetes, target_col="Outcome")

# Liver
liver = pd.read_csv("indian_liver_patient.csv")
train_model("Liver", liver, target_col="Dataset")

# Cancer
cancer = pd.read_csv("data.csv")
train_model("Cancer", cancer, target_col="diagnosis")

# Stroke
stroke = pd.read_csv("healthcare-dataset-stroke-data.csv")
train_model("Stroke", stroke, target_col="stroke")

# Hypertension
hypertension = pd.read_csv("framingham.csv")
train_model("Hypertension", hypertension, target_col="TenYearCHD")

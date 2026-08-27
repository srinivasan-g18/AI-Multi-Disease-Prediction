import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name=""):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"\n===== {model_name.upper()} MODEL =====")
    # Train metrics
    print("\n--- Train Metrics ---")
    print("Accuracy   :", accuracy_score(y_train, y_train_pred))
    print("Precision  :", precision_score(y_train, y_train_pred, average="weighted", zero_division=0))
    print("Recall     :", recall_score(y_train, y_train_pred, average="weighted", zero_division=0))
    print("F1-score   :", f1_score(y_train, y_train_pred, average="weighted", zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
    # Test metrics
    print("\n--- Test Metrics ---")
    print("Accuracy   :", accuracy_score(y_test, y_test_pred))
    print("Precision  :", precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    print("Recall     :", recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    print("F1-score   :", f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
def clean_data(df, target_col):
    for col in ["id", "ID", "Unnamed: 32"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

df = pd.read_csv("kidney_disease.csv")
target_col = "classification" if "classification" in df.columns else df.columns[-1]
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Kidney")

df = pd.read_csv("heart.csv")
possible_targets = ["target", "output", "Target", "num"]
target_col = next((col for col in possible_targets if col in df.columns), df.columns[-1])
X, y = clean_data(df, target_col)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
train_and_evaluate(rf_model, X_train, X_test, y_train, y_test, "Heart (RandomForest)")

svm_model = SVC(kernel="rbf", C=5, gamma="scale", class_weight="balanced", random_state=42)
train_and_evaluate(svm_model, X_train, X_test, y_train, y_test, "Heart (SVM)")

df = pd.read_csv("diabetes.csv")
target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Diabetes")

df = pd.read_csv("indian_liver_patient.csv")
target_col = "Dataset" if "Dataset" in df.columns else df.columns[-1]
if target_col in df.columns:
    df[target_col] = df[target_col].map({1: 0, 2: 1})  
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Liver")

df = pd.read_csv("data.csv")
target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Cancer")


df = pd.read_csv("healthcare-dataset-stroke-data.csv")
target_col = "stroke" if "stroke" in df.columns else df.columns[-1]
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Stroke")

df = pd.read_csv("framingham.csv")
target_col = "TenYearCHD" if "TenYearCHD" in df.columns else df.columns[-1]
X, y = clean_data(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_and_evaluate(RandomForestClassifier(random_state=42), X_train, X_test, y_train, y_test, "Hypertension")

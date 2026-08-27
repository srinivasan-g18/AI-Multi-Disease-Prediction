import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# 1. Load dataset
data = pd.read_csv("data.csv")  # your dataset

# 2. Drop unused columns
if "id" in data.columns:
    data = data.drop("id", axis=1)
if "Unnamed: 32" in data.columns:
    data = data.drop("Unnamed: 32", axis=1)

# 3. Handle missing values
data = data.fillna(data.median(numeric_only=True))

# 4. Encode target (M=1, B=0)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# 5. Keep only the 10 features you use in Streamlit
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",   # ✅ exactly as in dataset
    "symmetry_mean",
    "fractal_dimension_mean"
]

X = data[features]
y = data["diagnosis"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Pipeline with scaling
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000))
])

# 8. Train
model.fit(X_train, y_train)

# 9. Save model
with open("cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Cancer model trained with 10 features and saved as cancer_model.pkl")

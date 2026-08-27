import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# 1. Load dataset
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# 2. Handle missing values
data = data.fillna(data.median(numeric_only=True))

# 3. Encode categorical columns
for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# 4. Features and target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Pipeline with scaling
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

# 7. Train
model.fit(X_train, y_train)

# 8. Save model
with open("stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Stroke model trained and saved as stroke_model.pkl")

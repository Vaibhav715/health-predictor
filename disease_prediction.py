import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ================== DIABETES MODEL ==================

# Load dataset
diabetes = pd.read_csv("diabetes.csv")

# Columns with invalid 0 values
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Replace 0 with median
for col in cols:
    diabetes[col] = diabetes[col].replace(0, diabetes[col].median())

# Split features and target
X_d = diabetes.drop("Outcome", axis=1)
y_d = diabetes["Outcome"]

# Scaling
scaler = StandardScaler()
X_d = scaler.fit_transform(X_d)

# Train-test split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

# XGBoost Model (Optimized)
diabetes_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
diabetes_model.fit(X_train_d, y_train_d)

# Accuracy
y_pred_d = diabetes_model.predict(X_test_d)
print("XGBoost Diabetes Accuracy:", accuracy_score(y_test_d, y_pred_d))

# ================== PREDICTION FUNCTION ==================

def predict_diabetes(data):
    data = scaler.transform([data])   # VERY IMPORTANT
    return diabetes_model.predict(data)[0]

# ================== HEART MODEL ==================
heart = pd.read_csv("heart.csv")

X_h = heart.drop("target", axis=1)
y_h = heart["target"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

heart_model = RandomForestClassifier()
heart_model.fit(X_train_h, y_train_h)

# Accuracy (optional)
y_pred_h = heart_model.predict(X_test_h)
print("Heart Accuracy:", accuracy_score(y_test_h, y_pred_h))


# ================== FUNCTIONS ==================

def predict_diabetes(data):
    return diabetes_model.predict([data])[0]

def predict_heart(data):
    return heart_model.predict([data])[0]

# -------- Diabetes Probability --------
def predict_diabetes_proba(data):
    data = scaler.transform([data])
    return diabetes_model.predict_proba(data)[0][1]


# -------- Heart Probability --------
def predict_heart_proba(data):
    return heart_model.predict_proba([data])[0][1]
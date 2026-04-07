import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("air_quality_dataset.csv")

# =========================
# FEATURE ENGINEERING
# =========================
df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity'] / 100
df['Air_Stress'] = df['AirQuality'] / 500
df['Pressure_Deviation'] = abs(df['Pressure'] - 1013)

# Time-series features
df['AirQuality_MA'] = df['AirQuality'].rolling(window=5).mean()
df['AirQuality_Change'] = df['AirQuality'].diff()

# Interaction features (ADVANCED 🔥)
df['Gas_Temp_Interaction'] = df['AirQuality'] * df['Temperature']
df['Humidity_Pressure'] = df['Humidity'] * df['Pressure_Deviation']

# Handle NaN (latest pandas fix)
df = df.ffill().bfill()

# =========================
# ADVANCED RISK SCORING
# =========================
def risk_score(row):
    score = 0
    score += row['AirQuality'] * 0.4
    score += row['Temperature'] * 0.2
    score += row['Humidity'] * 0.2
    score += row['Pressure_Deviation'] * 0.2
    return score

df['Score'] = df.apply(risk_score, axis=1)

def classify(score):
    if score > 200:
        return "High"
    elif score > 120:
        return "Moderate"
    else:
        return "Safe"

df['Risk'] = df['Score'].apply(classify)

# =========================
# FEATURES & TARGET
# =========================
X = df[['Temperature', 'Humidity', 'AirQuality', 'Pressure',
        'Temp_Humidity_Index', 'Air_Stress',
        'Pressure_Deviation', 'AirQuality_MA', 'AirQuality_Change',
        'Gas_Temp_Interaction', 'Humidity_Pressure']]

y = df['Risk']

# Encode labels
y = y.map({"Safe": 0, "Moderate": 1, "High": 2})

# =========================
# FEATURE SCALING (IMPORTANT)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# XGBOOST MODEL
# =========================
model = XGBClassifier(
    objective='multi:softprob',  # needed for SHAP
    num_class=3,
    eval_metric='mlogloss'
)

params = {
    'n_estimators': [100],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(model, params, cv=3, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# =========================
# PREDICTION
# =========================
y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL + SCALER
# =========================
joblib.dump(best_model, "advanced_model_shap.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel & scaler saved ✅")

# =========================
# SHAP EXPLAINABILITY 🔥
# =========================
explainer = shap.Explainer(best_model)

# Convert back to DataFrame for SHAP
X_test_df = pd.DataFrame(X_test, columns=X.columns)

shap_values = explainer(X_test_df)

# =========================
# GLOBAL EXPLANATION
# =========================
print("\nGenerating SHAP summary plot...")

shap.summary_plot(shap_values, X_test_df)

# =========================
# LOCAL EXPLANATION (single prediction)
# =========================
print("\nGenerating SHAP force plot for one sample...")

shap.plots.waterfall(shap_values[0])
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("air_quality_dataset.csv")

# =========================
# SAFETY THRESHOLDS (based on real standards)
# WHO / ASHRAE / OSHA references
# =========================
THRESHOLDS = {
    # Temperature (°C): ASHRAE 55 comfort zone 18-26°C
    'temp_safe_min': 18,
    'temp_safe_max': 26,
    'temp_moderate_max': 40,
    'temp_danger': 50,
    'temp_physical_max': 60,    # above = physically impossible indoors → always High
    'temp_physical_min': -10,   # below = physically impossible indoors → always High

    # Humidity (%): ASHRAE 30-60% comfort; >80% mold risk; >95% danger
    'hum_safe_min': 30,
    'hum_safe_max': 60,
    'hum_moderate_max': 80,
    'hum_danger': 95,
    'hum_physical_max': 100,
    'hum_physical_min': 0,

    # Air Quality Index (0-500 scale)
    'aq_safe_max': 100,
    'aq_moderate_max': 200,
    'aq_danger': 300,
    'aq_physical_max': 500,
    'aq_physical_min': 0,

    # Pressure (hPa): Normal ~1013
    'pressure_moderate_dev': 15,
    'pressure_danger_dev': 30,
    'pressure_physical_max': 1100,
    'pressure_physical_min': 900,
}

# =========================
# FEATURE ENGINEERING
# =========================

# --- Normalized individual stress scores (each 0–1) ---
df['Temp_Stress'] = np.clip(
    (df['Temperature'] - THRESHOLDS['temp_safe_max']) /
    (THRESHOLDS['temp_danger'] - THRESHOLDS['temp_safe_max']), 0, 1
)

df['Humidity_Stress'] = np.where(
    df['Humidity'] > THRESHOLDS['hum_safe_max'],
    np.clip(
        (df['Humidity'] - THRESHOLDS['hum_safe_max']) /
        (THRESHOLDS['hum_danger'] - THRESHOLDS['hum_safe_max']), 0, 1
    ),
    np.where(
        df['Humidity'] < THRESHOLDS['hum_safe_min'],
        np.clip(
            (THRESHOLDS['hum_safe_min'] - df['Humidity']) / THRESHOLDS['hum_safe_min'], 0, 1
        ),
        0  # within safe range
    )
)

df['AQ_Stress'] = np.clip(
    df['AirQuality'] / THRESHOLDS['aq_danger'], 0, 1
)

df['Pressure_Dev'] = abs(df['Pressure'] - 1013)
df['Pressure_Stress'] = np.clip(
    df['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev'], 0, 1
)

# --- Raw ratio features (NOT clipped) — lets model see extreme out-of-range values ---
# e.g. 1000°C → Temp_Raw_Ratio = 1000/60 = 16.7  (vs 51°C = 0.85, clearly different)
df['Temp_Raw_Ratio']     = df['Temperature'] / THRESHOLDS['temp_physical_max']
df['Humidity_Raw_Ratio'] = df['Humidity']    / THRESHOLDS['hum_physical_max']
df['AQ_Raw_Ratio']       = df['AirQuality']  / THRESHOLDS['aq_physical_max']
df['Pressure_Raw_Ratio'] = abs(df['Pressure'] - 1013) / THRESHOLDS['pressure_danger_dev']

# --- Interaction features ---
df['Temp_Humidity_Discomfort'] = df['Temp_Stress'] * df['Humidity_Stress']
df['AQ_Temp_Interaction'] = df['AQ_Stress'] * df['Temp_Stress']
df['Combined_Environmental_Load'] = (
    df['Temp_Stress'] * 0.30 +
    df['Humidity_Stress'] * 0.20 +
    df['AQ_Stress'] * 0.35 +
    df['Pressure_Stress'] * 0.15
)  # weights sum to 1.0 ✅

# --- Raw features kept for model context ---
df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity'] / 100

df = df.ffill().bfill()

# =========================
# LABELING — Deterministic, threshold-based, priority-ordered
# Hard rules first (safety-critical), composite score as tiebreaker
# =========================
def classify(row):
    # ─── PHYSICALLY IMPOSSIBLE / OUT-OF-RANGE VALUES → always High ──
    if not (THRESHOLDS['temp_physical_min'] <= row['Temperature'] <= THRESHOLDS['temp_physical_max']):
        return "High"
    if not (THRESHOLDS['hum_physical_min']  <= row['Humidity']    <= THRESHOLDS['hum_physical_max']):
        return "High"
    if not (THRESHOLDS['aq_physical_min']   <= row['AirQuality']  <= THRESHOLDS['aq_physical_max']):
        return "High"
    if not (THRESHOLDS['pressure_physical_min'] <= row['Pressure'] <= THRESHOLDS['pressure_physical_max']):
        return "High"

    # ─── HARD DANGER RULES (any single factor critical) ─────────────
    if row['Temperature'] > THRESHOLDS['temp_danger']:
        return "High"
    if row['AirQuality'] > THRESHOLDS['aq_danger']:
        return "High"
    if row['Humidity'] > THRESHOLDS['hum_danger']:
        return "High"
    if row['Pressure_Dev'] > THRESHOLDS['pressure_danger_dev']:
        return "High"

    # ─── HARD MODERATE RULES ────────────────────────────────────────
    if row['Temperature'] > THRESHOLDS['temp_moderate_max']:
        return "Moderate"
    if row['AirQuality'] > THRESHOLDS['aq_moderate_max']:
        return "Moderate"
    if row['Humidity'] > THRESHOLDS['hum_moderate_max']:
        return "Moderate"
    if row['Pressure_Dev'] > THRESHOLDS['pressure_moderate_dev']:
        return "Moderate"

    # ─── COMPOSITE SCORE TIEBREAKER ─────────────────────────────────
    # Only reached if all individual factors are within safe ranges
    score = row['Combined_Environmental_Load']
    if score > 0.60:
        return "High"
    elif score > 0.30:
        return "Moderate"
    else:
        return "Safe"

df['Risk'] = df.apply(classify, axis=1)

# =========================
# INJECT SYNTHETIC EXTREME SAMPLES
# Real datasets rarely contain values like 1000°C or 200% humidity.
# Without training examples of these, the model never learns they are High.
# We inject a diverse set of extreme/impossible readings so the model
# sees clear signal in the Raw_Ratio features for out-of-range inputs.
# =========================
def make_extreme_row(temp, humidity, aq, pressure):
    """Build a fully-featured row for an extreme reading."""
    row = {
        'Temperature': temp,
        'Humidity':    humidity,
        'AirQuality':  aq,
        'Pressure':    pressure,
    }
    # Stress scores (clipped — same as training)
    row['Temp_Stress'] = float(np.clip(
        (temp - THRESHOLDS['temp_safe_max']) /
        (THRESHOLDS['temp_danger'] - THRESHOLDS['temp_safe_max']), 0, 1))

    hum_high = max(0.0, (humidity - THRESHOLDS['hum_safe_max']) /
                   (THRESHOLDS['hum_danger'] - THRESHOLDS['hum_safe_max']))
    hum_low  = max(0.0, (THRESHOLDS['hum_safe_min'] - humidity) / THRESHOLDS['hum_safe_min'])
    row['Humidity_Stress'] = float(min(max(hum_high, hum_low), 1))

    row['AQ_Stress']       = float(np.clip(aq / THRESHOLDS['aq_danger'], 0, 1))
    row['Pressure_Dev']    = abs(pressure - 1013)
    row['Pressure_Stress'] = float(np.clip(
        row['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev'], 0, 1))

    row['Temp_Humidity_Discomfort']    = row['Temp_Stress'] * row['Humidity_Stress']
    row['AQ_Temp_Interaction']         = row['AQ_Stress']   * row['Temp_Stress']
    row['Combined_Environmental_Load'] = (
        row['Temp_Stress']     * 0.30 +
        row['Humidity_Stress'] * 0.20 +
        row['AQ_Stress']       * 0.35 +
        row['Pressure_Stress'] * 0.15
    )
    row['Temp_Humidity_Index'] = temp * humidity / 100

    # Raw ratios (NOT clipped) — these carry the extreme signal
    row['Temp_Raw_Ratio']     = temp     / THRESHOLDS['temp_physical_max']
    row['Humidity_Raw_Ratio'] = humidity / THRESHOLDS['hum_physical_max']
    row['AQ_Raw_Ratio']       = aq       / THRESHOLDS['aq_physical_max']
    row['Pressure_Raw_Ratio'] = row['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev']

    # Label using the same classify logic
    row['Risk'] = classify(pd.Series(row))
    return row

extreme_cases = [
    # Extreme temperature
    (500,  50,  80, 1013), (1000, 50,  80, 1013), (200, 50,  80, 1013),
    (100,  50,  80, 1013), (80,   50,  80, 1013),  (-50, 50,  80, 1013),
    (-100, 50,  80, 1013),
    # Extreme humidity
    (22, 150,  80, 1013), (22, 200,  80, 1013), (22, 110,  80, 1013),
    (22,  -10, 80, 1013), (22,  -50, 80, 1013),
    # Extreme air quality
    (22, 50, 600,  1013), (22, 50, 1000, 1013), (22, 50, 800,  1013),
    (22, 50, 400,  1013), (22, 50, -10,  1013),
    # Extreme pressure
    (22, 50, 80, 800), (22, 50, 80, 700), (22, 50, 80, 1200),
    (22, 50, 80, 500), (22, 50, 80, 1150),
    # Multiple extremes at once
    (1000, 200, 1000, 500), (500, 150, 600, 700),
]

synthetic_rows = [make_extreme_row(t, h, a, p) for t, h, a, p in extreme_cases]
df_synthetic = pd.DataFrame(synthetic_rows)

# Repeat each synthetic sample 20x so they're not drowned out by real data
df_synthetic = pd.concat([df_synthetic] * 20, ignore_index=True)

df = pd.concat([df, df_synthetic], ignore_index=True)
print(f"\nInjected {len(df_synthetic)} synthetic extreme samples (all labeled High)")

# =========================
# CHECK DISTRIBUTION
# =========================
print("\nClass Distribution BEFORE balancing:")
print(df['Risk'].value_counts())

# =========================
# FEATURES & TARGET
# =========================
FEATURES = [
    'Temperature', 'Humidity', 'AirQuality', 'Pressure',
    'Temp_Stress', 'Humidity_Stress', 'AQ_Stress', 'Pressure_Stress',
    'Pressure_Dev', 'Temp_Humidity_Discomfort',
    'AQ_Temp_Interaction', 'Combined_Environmental_Load',
    'Temp_Humidity_Index',
    # Raw ratios (uncapped) — captures extreme/impossible values
    'Temp_Raw_Ratio', 'Humidity_Raw_Ratio', 'AQ_Raw_Ratio', 'Pressure_Raw_Ratio',
]

X = df[FEATURES]
y = df['Risk'].map({"Safe": 0, "Moderate": 1, "High": 2})

# =========================
# SPLIT BEFORE SCALING (prevent data leakage) ✅
# =========================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale fitted ONLY on training data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw), columns=FEATURES
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw), columns=FEATURES  # transform only, no fit
)

y_train_raw = y_train_raw.reset_index(drop=True)

# =========================
# BALANCING (on training set only) ✅
# =========================
train_data = pd.concat([X_train_scaled, y_train_raw], axis=1)

safe_df     = train_data[train_data['Risk'] == 0]
moderate_df = train_data[train_data['Risk'] == 1]
high_df     = train_data[train_data['Risk'] == 2]

n_target = len(safe_df)

moderate_up = resample(moderate_df, replace=True, n_samples=n_target, random_state=42)
high_up     = resample(high_df,     replace=True, n_samples=n_target, random_state=42)

balanced = pd.concat([safe_df, moderate_up, high_up]).sample(frac=1, random_state=42)

print("\nClass Distribution AFTER balancing:")
print(balanced['Risk'].value_counts())

X_bal = balanced.drop('Risk', axis=1)
y_bal = balanced['Risk'].astype(int)

# =========================
# MODEL with Stratified K-Fold CV ✅
# =========================
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False
)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 3],   # ← reduces overfitting
    'subsample': [0.8, 1.0],      # ← adds regularization
    'colsample_bytree': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(model, params, cv=cv, scoring='f1_macro', verbose=1, n_jobs=-1)
grid.fit(X_bal, y_bal)

best_model = grid.best_estimator_
print("\nBest Parameters:", grid.best_params_)

# =========================
# EVALUATION on held-out test set
# =========================
y_pred = best_model.predict(X_test_scaled)

label_map = {0: "Safe", 1: "Moderate", 2: "High"}

print("\nAccuracy:", accuracy_score(y_test_raw, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test_raw, y_pred,
    target_names=["Safe", "Moderate", "High"]
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_raw, y_pred))

# =========================
# FEATURE IMPORTANCE (built-in)
# =========================
importances = pd.Series(best_model.feature_importances_, index=FEATURES)
print("\nTop Feature Importances:")
print(importances.sort_values(ascending=False).head(10))

# =========================
# SAVE
# =========================
joblib.dump(best_model, "advanced_model_shap.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_list.pkl")  # save feature order too ✅

print("\nModel, scaler & feature list saved ✅")

# =========================
# SHAP EXPLANATIONS
# =========================
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_scaled)

print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test_scaled, class_names=["Safe", "Moderate", "High"])

print("\nGenerating SHAP waterfall plot (first test sample)...")
shap.plots.waterfall(shap_values[0])

# =========================
# PREDICT NEW SAMPLE (usage example)
# =========================
def predict_room_safety(temperature, humidity, air_quality, pressure):
    """
    Predict room safety for a single reading.
    Returns: (label, probabilities)
    """
    sample = {
        'Temperature': temperature,
        'Humidity': humidity,
        'AirQuality': air_quality,
        'Pressure': pressure,
    }

    # Replicate feature engineering
    sample['Temp_Stress'] = np.clip(
        (temperature - THRESHOLDS['temp_safe_max']) /
        (THRESHOLDS['temp_danger'] - THRESHOLDS['temp_safe_max']), 0, 1
    )
    hum_high = max(0, (humidity - THRESHOLDS['hum_safe_max']) /
                   (THRESHOLDS['hum_danger'] - THRESHOLDS['hum_safe_max']))
    hum_low  = max(0, (THRESHOLDS['hum_safe_min'] - humidity) / THRESHOLDS['hum_safe_min'])
    sample['Humidity_Stress'] = min(max(hum_high, hum_low), 1)
    sample['AQ_Stress'] = np.clip(air_quality / THRESHOLDS['aq_danger'], 0, 1)
    sample['Pressure_Dev'] = abs(pressure - 1013)
    sample['Pressure_Stress'] = np.clip(sample['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev'], 0, 1)
    sample['Temp_Humidity_Discomfort'] = sample['Temp_Stress'] * sample['Humidity_Stress']
    sample['AQ_Temp_Interaction'] = sample['AQ_Stress'] * sample['Temp_Stress']
    sample['Combined_Environmental_Load'] = (
        sample['Temp_Stress'] * 0.30 +
        sample['Humidity_Stress'] * 0.20 +
        sample['AQ_Stress'] * 0.35 +
        sample['Pressure_Stress'] * 0.15
    )
    sample['Temp_Humidity_Index'] = temperature * humidity / 100
    sample['Temp_Raw_Ratio']     = temperature / THRESHOLDS['temp_physical_max']
    sample['Humidity_Raw_Ratio'] = humidity    / THRESHOLDS['hum_physical_max']
    sample['AQ_Raw_Ratio']       = air_quality / THRESHOLDS['aq_physical_max']
    sample['Pressure_Raw_Ratio'] = abs(pressure - 1013) / THRESHOLDS['pressure_danger_dev']

    df_sample = pd.DataFrame([sample])[FEATURES]
    scaled = scaler.transform(df_sample)
    pred = best_model.predict(scaled)[0]
    proba = best_model.predict_proba(scaled)[0]

    label = {0: "Safe", 1: "Moderate", 2: "High"}[pred]
    print(f"\nPrediction: {label}")
    print(f"Probabilities → Safe: {proba[0]:.2f}, Moderate: {proba[1]:.2f}, High: {proba[2]:.2f}")
    return label, proba

# Example call:
# predict_room_safety(temperature=35, humidity=70, air_quality=180, pressure=1010)
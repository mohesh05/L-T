import joblib
import pandas as pd

# Load model + scaler
model = joblib.load("advanced_model_shap.pkl")
scaler = joblib.load("scaler.pkl")

# Input test cases
test_cases = [
    [25, 50, 80, 1012],
    [30, 65, 180, 1010],
    [38, 85, 400, 1005],
    [22, 40, 60, 1020],
    [35, 75, 250, 1008]
]

# Create DataFrame
df = pd.DataFrame(test_cases, columns=["Temperature", "Humidity", "AirQuality", "Pressure"])

# =========================
# RECREATE FEATURES (IMPORTANT)
# =========================
df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity'] / 100
df['Air_Stress'] = df['AirQuality'] / 500
df['Pressure_Deviation'] = abs(df['Pressure'] - 1013)

# Time features (simple approximation for prediction)
df['AirQuality_MA'] = df['AirQuality']
df['AirQuality_Change'] = 0

# Interaction features
df['Gas_Temp_Interaction'] = df['AirQuality'] * df['Temperature']
df['Humidity_Pressure'] = df['Humidity'] * df['Pressure_Deviation']

# =========================
# SELECT SAME FEATURES
# =========================
X = df[['Temperature', 'Humidity', 'AirQuality', 'Pressure',
        'Temp_Humidity_Index', 'Air_Stress',
        'Pressure_Deviation', 'AirQuality_MA', 'AirQuality_Change',
        'Gas_Temp_Interaction', 'Humidity_Pressure']]

# =========================
# SCALE DATA
# =========================
X_scaled = scaler.transform(X)

# =========================
# PREDICT
# =========================
predictions = model.predict(X_scaled)

# Convert numeric to labels
label_map = {0: "Safe", 1: "Moderate", 2: "High"}

for i, pred in enumerate(predictions):
    print(f"Test Case {i+1}: {test_cases[i]} → {label_map[pred]}")
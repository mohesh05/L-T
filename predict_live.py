import requests
import pandas as pd
import numpy as np
import joblib
import time

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load("advanced_model_shap.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# THINGSPEAK CONFIG
# ==============================
CHANNEL_ID  = "3061124"
READ_API_KEY = "NHA61ZCUA7S5DM4W"
READ_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"

# ==============================
# THRESHOLDS (must match training exactly)
# ==============================
THRESHOLDS = {
    'temp_safe_max':      26,
    'temp_moderate_max':  40,
    'temp_danger':        50,
    'temp_physical_max':  60,
    'temp_physical_min': -10,
    'hum_safe_min':       30,
    'hum_safe_max':       60,
    'hum_moderate_max':   80,
    'hum_danger':         95,
    'hum_physical_max':  100,
    'hum_physical_min':    0,
    'aq_safe_max':       100,
    'aq_moderate_max':   200,
    'aq_danger':         300,
    'aq_physical_max':   500,
    'aq_physical_min':     0,
    'pressure_moderate_dev': 15,
    'pressure_danger_dev':   30,
    'pressure_physical_max': 1100,
    'pressure_physical_min':  900,
}

# ==============================
# FEATURE LIST (must match training exactly)
# ==============================
FEATURES = [
    'Temperature', 'Humidity', 'AirQuality', 'Pressure',
    'Temp_Stress', 'Humidity_Stress', 'AQ_Stress', 'Pressure_Stress',
    'Pressure_Dev', 'Temp_Humidity_Discomfort',
    'AQ_Temp_Interaction', 'Combined_Environmental_Load',
    'Temp_Humidity_Index',
    'Temp_Raw_Ratio', 'Humidity_Raw_Ratio', 'AQ_Raw_Ratio', 'Pressure_Raw_Ratio',
]

label_map = {0: "Safe", 1: "Moderate", 2: "High"}

# ==============================
# HARD SAFETY CHECK (runs before model)
# ==============================
def hard_safety_check(temp, hum, aq, pressure):
    """Returns (label, reason) if a hard rule fires, else (None, None)."""
    pdev = abs(pressure - 1013)

    # Physically impossible values
    if not (THRESHOLDS['temp_physical_min']     <= temp     <= THRESHOLDS['temp_physical_max']):
        return "High", f"Temperature {temp}°C outside valid range ({THRESHOLDS['temp_physical_min']}–{THRESHOLDS['temp_physical_max']}°C)"
    if not (THRESHOLDS['hum_physical_min']      <= hum      <= THRESHOLDS['hum_physical_max']):
        return "High", f"Humidity {hum}% outside valid range (0–100%)"
    if not (THRESHOLDS['aq_physical_min']       <= aq       <= THRESHOLDS['aq_physical_max']):
        return "High", f"Air Quality {aq} outside valid range (0–500)"
    if not (THRESHOLDS['pressure_physical_min'] <= pressure <= THRESHOLDS['pressure_physical_max']):
        return "High", f"Pressure {pressure} hPa outside valid range (900–1100 hPa)"

    # Danger thresholds
    if temp     > THRESHOLDS['temp_danger']:         return "High",     f"Temperature {temp}°C > danger limit ({THRESHOLDS['temp_danger']}°C)"
    if hum      > THRESHOLDS['hum_danger']:          return "High",     f"Humidity {hum}% > danger limit ({THRESHOLDS['hum_danger']}%)"
    if aq       > THRESHOLDS['aq_danger']:           return "High",     f"Air Quality {aq} > danger limit ({THRESHOLDS['aq_danger']})"
    if pdev     > THRESHOLDS['pressure_danger_dev']: return "High",     f"Pressure deviation {pdev:.1f} hPa > danger limit ({THRESHOLDS['pressure_danger_dev']} hPa)"

    # Moderate thresholds
    if temp     > THRESHOLDS['temp_moderate_max']:   return "Moderate", f"Temperature {temp}°C > moderate limit ({THRESHOLDS['temp_moderate_max']}°C)"
    if hum      > THRESHOLDS['hum_moderate_max']:    return "Moderate", f"Humidity {hum}% > moderate limit ({THRESHOLDS['hum_moderate_max']}%)"
    if aq       > THRESHOLDS['aq_moderate_max']:     return "Moderate", f"Air Quality {aq} > moderate limit ({THRESHOLDS['aq_moderate_max']})"
    if pdev     > THRESHOLDS['pressure_moderate_dev']: return "Moderate", f"Pressure deviation {pdev:.1f} hPa > moderate limit ({THRESHOLDS['pressure_moderate_dev']} hPa)"

    return None, None  # let the model decide

# ==============================
# FEATURE ENGINEERING (must match training exactly)
# ==============================
def build_features(temp, hum, aq, pressure):
    row = {}
    row['Temperature'] = temp
    row['Humidity']    = hum
    row['AirQuality']  = aq
    row['Pressure']    = pressure

    # Stress scores (clipped 0–1)
    row['Temp_Stress'] = float(np.clip(
        (temp - THRESHOLDS['temp_safe_max']) /
        (THRESHOLDS['temp_danger'] - THRESHOLDS['temp_safe_max']), 0, 1))

    hum_high = max(0.0, (hum - THRESHOLDS['hum_safe_max']) /
                   (THRESHOLDS['hum_danger'] - THRESHOLDS['hum_safe_max']))
    hum_low  = max(0.0, (THRESHOLDS['hum_safe_min'] - hum) / THRESHOLDS['hum_safe_min'])
    row['Humidity_Stress'] = float(min(max(hum_high, hum_low), 1))

    row['AQ_Stress']       = float(np.clip(aq / THRESHOLDS['aq_danger'], 0, 1))
    row['Pressure_Dev']    = abs(pressure - 1013)
    row['Pressure_Stress'] = float(np.clip(
        row['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev'], 0, 1))

    # Interactions
    row['Temp_Humidity_Discomfort']    = row['Temp_Stress'] * row['Humidity_Stress']
    row['AQ_Temp_Interaction']         = row['AQ_Stress']   * row['Temp_Stress']
    row['Combined_Environmental_Load'] = (
        row['Temp_Stress']     * 0.30 +
        row['Humidity_Stress'] * 0.20 +
        row['AQ_Stress']       * 0.35 +
        row['Pressure_Stress'] * 0.15
    )

    row['Temp_Humidity_Index'] = temp * hum / 100

    # Raw ratios (uncapped — carry signal for extreme values)
    row['Temp_Raw_Ratio']     = temp     / THRESHOLDS['temp_physical_max']
    row['Humidity_Raw_Ratio'] = hum      / THRESHOLDS['hum_physical_max']
    row['AQ_Raw_Ratio']       = aq       / THRESHOLDS['aq_physical_max']
    row['Pressure_Raw_Ratio'] = row['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev']

    return row

# ==============================
# RISK EMOJI
# ==============================
def risk_emoji(label):
    return {"Safe": "✅", "Moderate": "⚠️", "High": "🚨"}.get(label, "❓")

# ==============================
# CONTINUOUS LOOP
# ==============================
print("🚀 Starting live room safety monitor... (Ctrl+C to stop)\n")

while True:
    try:
        response = requests.get(READ_URL, timeout=10)
        data = response.json()
        feed = data['feeds'][0]

        temp  = float(feed['field1'])
        hum   = float(feed['field2'])
        aq    = float(feed['field3'])
        press = float(feed['field4'])

        print("=" * 40)
        print("📡 LIVE SENSOR DATA")
        print(f"  🌡  Temperature : {temp} °C")
        print(f"  💧  Humidity    : {hum} %")
        print(f"  🌫  Air Quality : {aq}")
        print(f"  🌍  Pressure    : {press} hPa")

        # Step 1: hard safety check
        label, reason = hard_safety_check(temp, hum, aq, press)

        if label:
            print(f"\n  {risk_emoji(label)} Risk: {label}  (rule: {reason})")
        else:
            # Step 2: model prediction
            row    = build_features(temp, hum, aq, press)
            df_row = pd.DataFrame([row])[FEATURES]
            X_scaled = scaler.transform(df_row)
            pred   = model.predict(X_scaled)[0]
            proba  = model.predict_proba(X_scaled)[0]
            label  = label_map[pred]

            print(f"\n  {risk_emoji(label)} Risk: {label}")
            print(f"  📊 Confidence → Safe:{proba[0]:.0%}  Moderate:{proba[1]:.0%}  High:{proba[2]:.0%}")

    except Exception as e:
        print(f"⚠️  Error: {e}")

    time.sleep(15)
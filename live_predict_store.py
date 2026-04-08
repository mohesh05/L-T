import requests
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ===== LOAD MODEL =====
model = joblib.load("advanced_model_shap.pkl")
scaler = joblib.load("scaler.pkl")

# ===== THINGSPEAK =====
CHANNEL_ID   = "3061124"
READ_API_KEY = "NHA61ZCUA7S5DM4W"
URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"

# ===== CSV FILE =====
CSV_FILE = "live_air_quality_data.csv"
CSV_COLUMNS = ["Timestamp", "Temperature", "Humidity", "AirQuality", "Pressure",
               "Prediction", "Rule_Fired", "Safe%", "Moderate%", "High%"]

try:
    pd.read_csv(CSV_FILE)
except FileNotFoundError:
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_FILE, index=False)

# ===== THRESHOLDS (must match training exactly) =====
THRESHOLDS = {
    'temp_safe_max':         26,
    'temp_moderate_max':     40,
    'temp_danger':           50,
    'temp_physical_max':     60,
    'temp_physical_min':    -10,
    'hum_safe_min':          30,
    'hum_safe_max':          60,
    'hum_moderate_max':      80,
    'hum_danger':            95,
    'hum_physical_max':     100,
    'hum_physical_min':       0,
    'aq_safe_max':          100,
    'aq_moderate_max':      200,
    'aq_danger':            300,
    'aq_physical_max':      500,
    'aq_physical_min':        0,
    'pressure_moderate_dev': 15,
    'pressure_danger_dev':   30,
    'pressure_physical_max': 1100,
    'pressure_physical_min':  900,
}

FEATURES = [
    'Temperature', 'Humidity', 'AirQuality', 'Pressure',
    'Temp_Stress', 'Humidity_Stress', 'AQ_Stress', 'Pressure_Stress',
    'Pressure_Dev', 'Temp_Humidity_Discomfort',
    'AQ_Temp_Interaction', 'Combined_Environmental_Load',
    'Temp_Humidity_Index',
    'Temp_Raw_Ratio', 'Humidity_Raw_Ratio', 'AQ_Raw_Ratio', 'Pressure_Raw_Ratio',
]

label_map   = {0: "Safe", 1: "Moderate", 2: "High"}
risk_emoji  = {"Safe": "✅", "Moderate": "⚠️", "High": "🚨"}

# ===== HARD SAFETY CHECK =====
def hard_safety_check(temp, hum, aq, pressure):
    pdev = abs(pressure - 1013)
    if not (THRESHOLDS['temp_physical_min']     <= temp     <= THRESHOLDS['temp_physical_max']):
        return "High",     f"Temperature {temp}°C out of range"
    if not (THRESHOLDS['hum_physical_min']      <= hum      <= THRESHOLDS['hum_physical_max']):
        return "High",     f"Humidity {hum}% out of range"
    if not (THRESHOLDS['aq_physical_min']       <= aq       <= THRESHOLDS['aq_physical_max']):
        return "High",     f"AirQuality {aq} out of range"
    if not (THRESHOLDS['pressure_physical_min'] <= pressure <= THRESHOLDS['pressure_physical_max']):
        return "High",     f"Pressure {pressure} hPa out of range"
    if temp     > THRESHOLDS['temp_danger']:          return "High",     f"Temp {temp}°C > {THRESHOLDS['temp_danger']}°C"
    if hum      > THRESHOLDS['hum_danger']:           return "High",     f"Humidity {hum}% > {THRESHOLDS['hum_danger']}%"
    if aq       > THRESHOLDS['aq_danger']:            return "High",     f"AQ {aq} > {THRESHOLDS['aq_danger']}"
    if pdev     > THRESHOLDS['pressure_danger_dev']:  return "High",     f"Pressure dev {pdev:.1f} > {THRESHOLDS['pressure_danger_dev']} hPa"
    if temp     > THRESHOLDS['temp_moderate_max']:    return "Moderate", f"Temp {temp}°C > {THRESHOLDS['temp_moderate_max']}°C"
    if hum      > THRESHOLDS['hum_moderate_max']:     return "Moderate", f"Humidity {hum}% > {THRESHOLDS['hum_moderate_max']}%"
    if aq       > THRESHOLDS['aq_moderate_max']:      return "Moderate", f"AQ {aq} > {THRESHOLDS['aq_moderate_max']}"
    if pdev     > THRESHOLDS['pressure_moderate_dev']: return "Moderate", f"Pressure dev {pdev:.1f} > {THRESHOLDS['pressure_moderate_dev']} hPa"
    return None, None

# ===== FEATURE ENGINEERING =====
def build_features(temp, hum, aq, pressure):
    row = {}
    row['Temperature'] = temp
    row['Humidity']    = hum
    row['AirQuality']  = aq
    row['Pressure']    = pressure

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

    row['Temp_Humidity_Discomfort']    = row['Temp_Stress'] * row['Humidity_Stress']
    row['AQ_Temp_Interaction']         = row['AQ_Stress']   * row['Temp_Stress']
    row['Combined_Environmental_Load'] = (
        row['Temp_Stress']     * 0.30 +
        row['Humidity_Stress'] * 0.20 +
        row['AQ_Stress']       * 0.35 +
        row['Pressure_Stress'] * 0.15
    )
    row['Temp_Humidity_Index']  = temp * hum / 100
    row['Temp_Raw_Ratio']       = temp     / THRESHOLDS['temp_physical_max']
    row['Humidity_Raw_Ratio']   = hum      / THRESHOLDS['hum_physical_max']
    row['AQ_Raw_Ratio']         = aq       / THRESHOLDS['aq_physical_max']
    row['Pressure_Raw_Ratio']   = row['Pressure_Dev'] / THRESHOLDS['pressure_danger_dev']
    return row

# ===== MAIN LOOP =====
print("🚀 Live Monitoring Started...\n")

while True:
    try:
        response = requests.get(URL, timeout=10)
        data     = response.json()
        feed     = data['feeds'][0]

        temp  = float(feed['field1'])
        hum   = float(feed['field2'])
        aq    = float(feed['field3'])
        press = float(feed['field4'])
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("=" * 40)
        print(f"📡 {ts}")
        print(f"  🌡  Temperature : {temp} °C")
        print(f"  💧  Humidity    : {hum} %")
        print(f"  🌫  AirQuality  : {aq}")
        print(f"  🌍  Pressure    : {press} hPa")

        # Step 1 — hard rule check
        label, reason = hard_safety_check(temp, hum, aq, press)
        rule_fired = reason if reason else "None"
        safe_p = mod_p = high_p = "-"

        if label:
            print(f"\n  {risk_emoji[label]} Prediction : {label}")
            print(f"  📋 Rule fired : {reason}")
        else:
            # Step 2 — model prediction
            row      = build_features(temp, hum, aq, press)
            df_row   = pd.DataFrame([row])[FEATURES]
            X_scaled = scaler.transform(df_row)
            pred     = model.predict(X_scaled)[0]
            proba    = model.predict_proba(X_scaled)[0]
            label    = label_map[pred]
            safe_p   = f"{proba[0]:.0%}"
            mod_p    = f"{proba[1]:.0%}"
            high_p   = f"{proba[2]:.0%}"

            print(f"\n  {risk_emoji[label]} Prediction : {label}")
            print(f"  📊 Confidence → Safe:{safe_p}  Moderate:{mod_p}  High:{high_p}")

        # Save to CSV
        row_csv = pd.DataFrame([[ts, temp, hum, aq, press,
                                  label, rule_fired, safe_p, mod_p, high_p]],
                                columns=CSV_COLUMNS)
        row_csv.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"\n  💾 Saved to {CSV_FILE} ✅")

    except Exception as e:
        print(f"❌ Error: {e}")

    time.sleep(15)
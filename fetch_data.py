import requests
import pandas as pd
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
CHANNEL_ID = "3061124"
READ_API_KEY = "NHA61ZCUA7S5DM4W"

READ_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"

# ==============================
# LABEL MAP
# ==============================
label_map = {
    0: "Safe",
    1: "Moderate",
    2: "High"
}

# ==============================
# CONTINUOUS LOOP
# ==============================
while True:
    try:
        # ===== FETCH DATA =====
        response = requests.get(READ_URL)
        data = response.json()
        feed = data['feeds'][0]

        temp = float(feed['field1'])
        hum = float(feed['field2'])
        air = float(feed['field3'])
        press = float(feed['field4'])

        # ===== PRINT SENSOR VALUES =====
        print("\n==============================")
        print("📡 LIVE SENSOR DATA")
        print(f"🌡 Temperature : {temp} °C")
        print(f"💧 Humidity    : {hum} %")
        print(f"🌫 Air Quality : {air}")
        print(f"🌍 Pressure    : {press} hPa")

        # ===== CREATE DATAFRAME =====
        df = pd.DataFrame([[temp, hum, air, press]],
                          columns=["Temperature", "Humidity", "AirQuality", "Pressure"])

        # ===== FEATURE ENGINEERING =====
        df['Temp_Humidity_Index'] = df['Temperature'] * df['Humidity'] / 100
        df['Air_Stress'] = df['AirQuality'] / 500
        df['Pressure_Deviation'] = abs(df['Pressure'] - 1013)

        df['AirQuality_MA'] = df['AirQuality']
        df['AirQuality_Change'] = 0

        df['Gas_Temp_Interaction'] = df['AirQuality'] * df['Temperature']
        df['Humidity_Pressure'] = df['Humidity'] * df['Pressure_Deviation']

        # ===== SELECT FEATURES =====
        X = df[['Temperature', 'Humidity', 'AirQuality', 'Pressure',
                'Temp_Humidity_Index', 'Air_Stress',
                'Pressure_Deviation', 'AirQuality_MA', 'AirQuality_Change',
                'Gas_Temp_Interaction', 'Humidity_Pressure']]

        # ===== SCALE =====
        X_scaled = scaler.transform(X)

        # ===== PREDICT =====
        pred = model.predict(X_scaled)[0]
        result = label_map[pred]

        # ===== PRINT PREDICTION =====
        print(f"🧠 Predicted Risk: {result}")

    except Exception as e:
        print("⚠️ Error:", e)

    # ===== WAIT (ThingSpeak limit) =====
    time.sleep(15)
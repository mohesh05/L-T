import pandas as pd
import numpy as np

np.random.seed(42)
rows = 20000

# =========================
# TIME SIMULATION
# =========================
time_index = pd.date_range(start="2024-01-01", periods=rows, freq="h")
hour = time_index.hour

# =========================
# TEMPERATURE (day-night cycle)
# =========================
temperature = 25 + 10 * np.sin((hour / 24) * 2 * np.pi) + np.random.normal(0, 2, rows)

# =========================
# HUMIDITY (inverse relation with temperature)
# =========================
humidity = 80 - (temperature - 25) * 1.5 + np.random.normal(0, 5, rows)

# =========================
# AIR QUALITY (normal + spikes)
# =========================
air_quality = np.random.normal(120, 40, rows)

# Pollution spikes
spike_indices = np.random.choice(rows, size=int(rows * 0.1))
air_quality[spike_indices] += np.random.uniform(150, 300, len(spike_indices))

air_quality = np.clip(air_quality, 50, 500)

# =========================
# PRESSURE (realistic variation)
# =========================
pressure = 1013 + np.random.normal(0, 5, rows)

# =========================
# CREATE DATAFRAME
# =========================
df = pd.DataFrame({
    "Timestamp": time_index,
    "Temperature": temperature,
    "Humidity": humidity,
    "AirQuality": air_quality,
    "Pressure": pressure
})

# =========================
# SENSOR NOISE (realistic)
# =========================
df['Temperature'] += np.random.normal(0, 1.5, len(df))
df['Humidity'] += np.random.normal(0, 3, len(df))
df['AirQuality'] += np.random.normal(0, 10, len(df))
df['Pressure'] += np.random.normal(0, 2, len(df))

# =========================
# CLAMP VALUES (important)
# =========================
df['Temperature'] = df['Temperature'].clip(15, 55)
df['Humidity'] = df['Humidity'].clip(20, 100)
df['AirQuality'] = df['AirQuality'].clip(50, 500)
df['Pressure'] = df['Pressure'].clip(980, 1035)

# =========================
# 🔥 BALANCED RISK LOGIC
# =========================
def classify_risk(row):

    # Normalize all features (0–1 range)
    aq_norm = row['AirQuality'] / 500
    temp_norm = (row['Temperature'] - 15) / 40
    hum_norm = row['Humidity'] / 100
    press_norm = abs(row['Pressure'] - 1013) / 25

    # Base score
    score = (
        aq_norm * 0.35 +
        temp_norm * 0.25 +
        hum_norm * 0.2 +
        press_norm * 0.2
    )

    # Interaction effects 🔥
    score += (aq_norm * temp_norm) * 0.2
    score += (hum_norm * press_norm) * 0.1

    # Extreme conditions
    if row['AirQuality'] > 350:
        score += 0.15
    if row['Temperature'] > 42:
        score += 0.1
    if row['Humidity'] > 90:
        score += 0.05
    if abs(row['Pressure'] - 1013) > 20:
        score += 0.1

    # Classification
    if score > 0.75:
        return "High"
    elif score > 0.45:
        return "Moderate"
    else:
        return "Safe"

df['Risk'] = df.apply(classify_risk, axis=1)

# =========================
# ADD LABEL NOISE (real-world behavior)
# =========================
noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)))
df.loc[noise_idx, 'Risk'] = np.random.choice(['Safe','Moderate','High'], len(noise_idx))

# =========================
# CHECK DISTRIBUTION
# =========================
print("\nClass Distribution:")
print(df['Risk'].value_counts())

# =========================
# SAVE DATASET
# =========================
df.to_csv("air_quality_dataset.csv", index=False)

print("\nDataset created successfully ✅")
print(df.head())
import pandas as pd
import numpy as np

np.random.seed(42)

rows = 200000

data = {
    "Temperature": np.random.uniform(20, 40, rows),
    "Humidity": np.random.uniform(30, 90, rows),
    "AirQuality": np.random.uniform(50, 500, rows),
    "Pressure": np.random.uniform(980, 1030, rows)
}

df = pd.DataFrame(data)

# Risk classification logic (REALISTIC)
def classify_risk(row):
    if row['AirQuality'] > 300 or row['Temperature'] > 35 or row['Humidity'] > 80:
        return "High"
    elif row['AirQuality'] > 150 or row['Temperature'] > 30 or row['Humidity'] > 60:
        return "Moderate"
    else:
        return "Safe"

df['Risk'] = df.apply(classify_risk, axis=1)

df.to_csv("air_quality_dataset.csv", index=False)

print("Dataset created successfully ✅")
print(df.head())
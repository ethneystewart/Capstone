import pandas as pd
from scipy.stats import zscore

# === 1. Create the dataset ===
data = {
    "date": [
        "2024-12-01", "2024-12-02", "2024-12-03", "2024-12-04", "2024-12-05",
        "2024-12-06", "2024-12-07", "2024-12-08", "2024-12-09", "2024-12-10",
        "2024-12-11", "2024-12-12", "2024-12-13", "2024-12-14", "2024-12-15",
        "2024-12-16", "2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20",
        "2024-12-21", "2024-12-22", "2024-12-23", "2024-12-24", "2024-12-25",
        "2024-12-26", "2024-12-27", "2024-12-28", "2024-12-29", "2024-12-30", "2024-12-31"
    ],
    "kWh": [
        2.670, 3.186, 4.160, 5.146, 6.527, 9.137, 10.580, 7.600, 6.269, 5.534,
        5.467, 12.531, 23.219, 9.980, 14.554, 12.248, 10.410, 10.922, 10.157,
        8.497, 10.507, 8.747, 10.199, 8.870, 8.173, 9.854, 10.738, 9.639,
        10.803, 10.574, 13.254
    ]
}

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# === 2. Calculate Z-scores ===
df["z_score"] = zscore(df["kWh"])

# === 3. Filter: keep only non-anomalous days (z-score < ±2) ===
filtered_df = df[df["z_score"].abs() < 2]

# === 4. Calculate typical daily kWh ===
typical_kwh = filtered_df["kWh"].mean()

# === 5. Show results ===
print("Typical (non-anomalous) average daily energy usage:", round(typical_kwh, 2), "kWh")
print("\nAnomalous days:")
print(df[df["z_score"].abs() >= 2][["date", "kWh", "z_score"]])

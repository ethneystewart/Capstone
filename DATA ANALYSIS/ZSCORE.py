import pandas as pd
from scipy.stats import zscore

# === 1. Load the data ===
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'Hourly Data'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# === 2. Convert 'date' to datetime (if needed) ===
df["date"] = pd.to_datetime(df["date"])

# === 3. Compute Z-scores for wave heights ===
df["z_score"] = zscore(df["wave_period"])

# === 4. Filter out outliers (use Z-score threshold, e.g., ±2) ===
filtered = df[df["z_score"].abs() < 2]

# === 5. Calculate typical wave height ===
typical_wave_period = filtered["wave_period"].mean()

# === 6. Output the result ===
print(f"Typical (non-anomalous) wave period: {typical_wave_period:.2f} meters")

import pandas as pd

# === 1. Load the data ===
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'Hourly Data'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# === 2. Prepare the data ===
df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.date

# === 3. Calculate daily average wave height ===
daily_avg = df.groupby("day")["wave_height"].mean()

# === 4. Compute IQR ===
Q1 = daily_avg.quantile(0.25)
Q3 = daily_avg.quantile(0.75)
IQR = Q3 - Q1

# === 5. Determine bounds for non-anomalous values ===
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# === 6. Filter out anomalous days ===
filtered_avg = daily_avg[(daily_avg >= lower_bound) & (daily_avg <= upper_bound)]

# === 7. Compute typical average wave height ===
typical_wave_height = filtered_avg.mean()

# === 8. Output result ===
print(f"Typical (non-anomalous) average wave height: {typical_wave_height:.2f} meters")

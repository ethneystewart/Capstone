import pandas as pd

# === 1. Load the data ===
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'Hourly Data'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# === 2. Convert 'date' to datetime if it's not already ===
df["date"] = pd.to_datetime(df["date"])

# === 3. Compute IQR on individual wave height values ===
Q1 = df["wave_height"].quantile(0.25)
Q3 = df["wave_height"].quantile(0.75)
IQR = Q3 - Q1

# === 4. Define bounds ===
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# === 5. Filter out anomalous wave heights ===
filtered = df[(df["wave_height"] >= lower_bound) & (df["wave_height"] <= upper_bound)]

# === 6. Calculate the typical wave height from filtered values ===
typical_wave_height = filtered["wave_height"].mean()

# === 7. Output result ===
print(f"Typical (non-anomalous) wave height: {typical_wave_height:.2f} meters")

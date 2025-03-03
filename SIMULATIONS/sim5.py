import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from tqdm import tqdm

# Define parameters for the simulation
rho = 1025          # Density of water (kg/m^3)
Cd = 0.6            # Drag coefficient
Cm = 1              # Added mass coefficient
A_buoy = 0.5        # Cross-sectional area of buoy (m^2)
V = 1               # Displaced volume of buoy (m^3)
m = 768.5           # Mass of buoy (kg)
A_coil = 0.1        # Coil area (m^2)
R = 72              # Electrical resistance (Ohms)

# Define constant wave frequency and amplitude
wave_freq = 0.5     # Wave frequency (Hz)
wave_amp = 1        # Wave amplitude (m)

# Load Excel file with data
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'Hourly Data'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    #print(df)
except FileNotFoundError:
    raise FileNotFoundError(f"Excel file not found: {file_path}")
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the Excel file: {e}")

# Validate required columns
required_columns = {'wave_height', 'wave_period', 'date'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing required columns in the dataset: {required_columns - set(df.columns)}")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Simulation parameters
depth_of_water = 27  # Depth of water (m)
k = 500              # Spring constant (N/m)
B = 1                # Magnetic field strength (T)
N = 50               # Number of coil turns
c = 20               # Damping coefficient (Ns/m)

dt = 0.5            # Time step (s)
sim_time = 3600      # Simulation time per hour in seconds (1 hour)

def simulate_hourly_power(wave_height, wave_period):
    time = np.arange(0, sim_time, dt)
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)

    omega = 2 * np.pi / wave_period
    amplitude = wave_height / 2
    for i in range(len(time) - 1):
        elapsed_time = time[i]
        wave_disp = amplitude * np.sin(omega * elapsed_time)
        wave_vel = amplitude * omega * np.cos(omega * elapsed_time)
        wave_acc = -amplitude * omega**2 * np.sin(omega * elapsed_time)

        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel - buoy_vel[i]) * abs(wave_vel - buoy_vel[i]) +
                     Cm * V * rho * (wave_acc))
        F_spring = k * buoy_disp[i]
        F_damp = c * buoy_vel[i]

        F_total = F_morison - F_spring - F_damp
        buoy_acc[i + 1] = F_total / m

        buoy_vel[i + 1] = buoy_vel[i] + buoy_acc[i + 1] * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        V_out = -N * B * A_coil * buoy_vel[i + 1]
        Pout[i + 1] = V_out**2 / R

    total_power_hour = np.sum(Pout) * dt / 3600
    return total_power_hour, np.max(amplitude) #this is amplitude why????

# Simulate for each hour and day
hourly_data = []
daily_summary = []

for day, group in tqdm(df.groupby(df['date'].dt.date)):
    total_power_day = 0
    max_wave_height_day = 0

    for _, row in group.iterrows():
        wave_height = row['wave_height']
        wave_period = row['wave_period']
        power_hour, max_wave_height = simulate_hourly_power(wave_height, wave_period)

        hourly_data.append({
            'date': row['date'],
            'wave_height': wave_height,
            'wave_period': wave_period,
            'power_output': power_hour
        })

        total_power_day += power_hour
        max_wave_height_day = max(max_wave_height_day, max_wave_height)

    daily_summary.append({
        'day': day,
        'total_power': total_power_day,
        'max_wave_height': max_wave_height_day
    })

# Convert results to DataFrames
hourly_data_df = pd.DataFrame(hourly_data)
daily_summary_df = pd.DataFrame(daily_summary)

# Calculate total power over the entire period
total_power = daily_summary_df['total_power'].sum()

# Save results to Excel
with pd.ExcelWriter('wave_power_results.xlsx') as writer:
    hourly_data_df.to_excel(writer, sheet_name='Hourly Data', index=False)
    daily_summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)

print(f"Total power over the entire period: {total_power} kWh")

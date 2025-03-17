import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# Simulation parameters
k = 500              # Spring constant (N/m)
B = 1                # Magnetic field strength (T)
N = 50               # Number of coil turns
c = 20               # Damping coefficient (Ns/m)

dt = 0.5            # Time step (s)
sim_time = 3600      # Simulation time per hour in seconds (1 hour)

# Load Excel file with data
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'DEC1'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
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


def jonswap_spectrum(f, Hs, Tp, gamma=3.3):
    """
    Compute the JONSWAP spectral density for a given frequency.

    Hs: Significant wave height (m)
    Tp: Peak wave period (s)
    gamma: Peak enhancement factor (typically 3.3)

    Returns spectral density S(f) at the given frequency f.
    """
    g = 9.81  # Gravity (m/s²)
    
    fp = 1 / Tp  # Peak frequency
    sigma = 0.07 if f <= fp else 0.09  # Bandwidth parameter

    alpha = 0.076 * (Hs**2) * (fp**4)  
    r = np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))  

    S = alpha * g**2 * f**(-5) * np.exp(-5/4 * (fp / f)**4) * gamma**r
    return S

def complex_wave_displacement(time, wave_height, wave_period):
    """
    Generate a simple wave motion using only the peak frequency of the JONSWAP spectrum.

    wave_height: Significant wave height (m)
    wave_period: Peak wave period (s)
    """
    fp = 1 / wave_period  # Peak frequency
    S_fp = jonswap_spectrum(fp, wave_height, wave_period)  # Spectrum at peak frequency
    
    # Estimate wave amplitude from significant wave height
    amplitude = wave_height / 2  # Approximate based on ocean wave theory
    
    # Assume zero initial phase for simplicity
    phase = 0  

    # Compute wave motion at peak frequency
    wave_disp = amplitude * np.sin(2 * np.pi * fp * time + phase)
    wave_vel = 2 * np.pi * fp * amplitude * np.cos(2 * np.pi * fp * time + phase)
    wave_acc = -(2 * np.pi * fp)**2 * amplitude * np.sin(2 * np.pi * fp * time + phase)

    return wave_disp, wave_vel, wave_acc, amplitude


def simulate_hourly_power(wave_height, wave_period):
    time = np.arange(0, sim_time, dt)
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)

    # Generate simple wave motion based on JONSWAP peak frequency
    wave_disp, wave_vel, wave_acc, max_amplitude = complex_wave_displacement(time, wave_height, wave_period)
    
    for i in range(len(time) - 1):
        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel[i] - buoy_vel[i]) * abs(wave_vel[i] - buoy_vel[i]) +
                     Cm * V * rho * wave_acc[i])
        F_spring = k * buoy_disp[i]
        F_damp = c * buoy_vel[i]

        F_total = F_morison - F_spring - F_damp
        
        buoy_acc[i + 1] = F_total / m
        buoy_vel[i + 1] = buoy_vel[i] + buoy_acc[i + 1] * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        V_out = -N * B * A_coil * buoy_vel[i + 1]
        Pout[i + 1] = V_out**2 / R

    total_power_hour = np.trapz(Pout, dx=dt) / 3600

    return total_power_hour, max_amplitude


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

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))

axes[0].plot(daily_summary_df['day'], daily_summary_df['total_power'], marker='o', linestyle='-', label='Total Power Output (kWh)')
axes[0].set_title('Daily Total Power Output', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0].grid(True)
axes[0].legend(fontsize=8)

axes[1].plot(daily_summary_df['day'], daily_summary_df['max_wave_height'], marker='s', linestyle='-', color='r', label='Max Wave Height (m)')
axes[1].set_title('Daily Maximum Wave Height', fontsize=10, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].grid(True)
axes[1].legend(fontsize=8)

axes[2].hist(hourly_data_df['power_output'], bins=30, color='g', alpha=0.7, edgecolor='black')
axes[2].set_title('Distribution of Hourly Power Output', fontsize=10, fontweight='bold')
axes[2].tick_params(axis='x', labelsize=8)
axes[2].grid(True)

plt.tight_layout()
plt.show()

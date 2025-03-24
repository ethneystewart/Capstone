import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp

# Define parameters for the simulation
rho = 1025          # Density of water (kg/m^3)
Cd = 0.6            # Drag coefficient
Cm = 1              # Added mass coefficient
D = 0.342951*2      # Diameter of buoy (m)
V = 0.647944        # Displaced volume of buoy (m^3)
m = 609.06736       # Mass of buoy (kg)
A_coil = 0.1        # Coil area (m^2)
R = 72              # Electrical resistance (Ohms)


# Simulation parameters
k = 500             # Spring constant (N/m)
B = 1               # Magnetic field strength (T)
N = 50              # Number of coil turns
c = 20              # Damping coefficient (Ns/m)
dt = 0.5            # Time step (s)
sim_time = 3600     # Simulation time per hour in seconds (1 hour)
time = np.arange(0, sim_time, dt)  # Define time array

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


def jonswap_spectrum(f, Hs, Tp, gamma=2):
    """
    Compute the JONSWAP spectral density for a given frequency.
    spectral density of ocean waves based on jonswap spectrum
    
    Hs: Significant wave height (m) - avg high of one-third of waves
    Tp: Peak wave period (s) - period at which max wave energy occurs 
    gamma: Peak enhancement factor (typically 3.3) - controls sharpness of the spectrum around the peak frequency

    returns spectral density S(f) at the given frequency f - which represents how wave energy  is distributed across different frequencies 
    """
    g = 9.81  # Gravity (m/s²)
    
    fp = 1 / Tp  # Peak frequency
    sigma = np.where(f <= fp, 0.07, 0.09)  # Bandwidth parameter (vectorized) - 0.07 for frequencies below peak and 0.09 for frequencies above peak 
    # asymetry accounts for fact that real ocean waves are not symmetrical around peak frequency

    alpha = 0.076 * (Hs**2) * (fp**4)  # Scale factor - determines magnitude of the spectrum - higher hs more wave energy
    r = np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))  # Peak enhancement - applies a gaussian shape centered around peak frequency
    
    S = alpha * g**2 * f**(-5) * np.exp(-5/4 * (fp / f)**4) * gamma**r # JONSWAP spectrum formula
    return S

def generate_wave_components(Hs, Tp, num_waves=10):
    """
    Generate multiple wave components based on the JONSWAP spectrum with proper amplitude scaling.
    """
    f_min, f_max = 0.5 / Tp, 2.5 / Tp  # Frequency range
    freqs = np.linspace(f_min, f_max, num_waves)  # Wave frequencies
    
    # Compute spectral density for each frequency
    S_f = jonswap_spectrum(freqs, Hs, Tp)
    
    # Correct amplitude estimation from significant wave height
    amplitudes = (Hs / np.sqrt(2)) * np.sqrt(S_f / S_f.sum())  
    """
    in ocean wave theory significant wave height is related to root mean sware wave height 
    this conversion was nescessary because wave amplitudes are normally distributed 

    second part takes square root of the normalized spectral density to adjust the amplitude to correctly match the wave energy distribution 
    """
    # Randomize phases
    phases = np.random.uniform(0, 2 * np.pi, num_waves)
    
    return freqs, amplitudes, phases

def complex_wave_displacement(time, wave_height, wave_period, num_waves=10):
    """
    Generate a realistic wave using JONSWAP spectrum-based superposition with corrected amplitude scaling.
    """
    freqs, amplitudes, phases = generate_wave_components(wave_height, wave_period, num_waves)
    
    # Compute displacement, velocity, and acceleration
    wave_disp = np.sum([
        amplitudes[i] * np.sin(2 * np.pi * freqs[i] * time + phases[i])
        for i in range(num_waves)
    ], axis=0)
    
    wave_vel = np.sum([
        2 * np.pi * freqs[i] * amplitudes[i] * np.cos(2 * np.pi * freqs[i] * time + phases[i])
        for i in range(num_waves)
    ], axis=0)
    
    wave_acc = np.sum([
        -(2 * np.pi * freqs[i])**2 * amplitudes[i] * np.sin(2 * np.pi * freqs[i] * time + phases[i])
        for i in range(num_waves)
    ], axis=0)

    # print("Any NaNs in wave_disp?", np.isnan(wave_disp).any())
    # print("Any NaNs in wave_vel?", np.isnan(wave_vel).any())
    # print("Any NaNs in wave_acc?", np.isnan(wave_acc).any())
    
    return wave_disp, wave_vel, wave_acc, np.max(amplitudes)


def simulate_hourly_power(wave_height, wave_period):
    time = np.arange(0, sim_time, dt)

    # Generate wave motion using your existing function
    wave_disp, wave_vel, wave_acc, max_amplitude = complex_wave_displacement(time, wave_height, wave_period)

    # Define the ODE system: state = [x, v]
    def system(t, y):
        x, v = y
        
        # Interpolate wave velocity and acceleration at time t
        wave_v = np.interp(t, time, wave_vel)
        wave_a = np.interp(t, time, wave_acc)
        
        # Calculate terms
        delta_v = wave_v - v
        linear_drag = 0.5 * rho * Cd * D * delta_v * abs(delta_v)
        inertia_force = rho * Cm ** (np.pi/4) * D**2 * V * wave_a
        
        # Denominator includes added mass
        denominator = m + rho * Cm * V
        
        # Total acceleration considering added mass effect
        a = (linear_drag + inertia_force - k * x - c * v) / denominator
        
        return [v, a]


    # Initial conditions: displacement and velocity
    y0 = [0.0, 0.0]

    # Solve the ODE system with RK45 (adaptive Runge-Kutta)
    sol = solve_ivp(system, [0, sim_time], y0, t_eval=time, method='RK45')

    # Extract displacement and velocity
    disp = sol.y[0]
    vel = sol.y[1]

    # Calculate output voltage and power
    V_out = -N * B * A_coil * vel
    Pout = V_out**2 / R

    # Integrate power over time to get total energy in Watt-hours
    total_power_hour = np.trapz(Pout, dx=dt) / 3600  # divide by 3600 to get Wh

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
        
        # Compute full wave displacement for this hour
        wave_disp, _, _, _ = complex_wave_displacement(time, wave_height, wave_period)
        
        power_hour, max_wave_height = simulate_hourly_power(wave_height, wave_period)

        # Store data for plotting
        hourly_data.append({
            'date': row['date'],
            'wave_height': wave_height,
            'wave_period': wave_period,
            'power_output': power_hour,
            'wave_displacement': wave_disp  # Store full wave displacement array
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

import matplotlib.dates as mdates

#this will output for a days worth of the power otput each hour and the inputted wave height from api 
# Create figure and subplots
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)  # 2 rows, 1 column

# 📊 Subplot 1: Power Output Over Time
ax[0].plot(hourly_data_df['date'], hourly_data_df['power_output'], label='Power Output (kWh)', color='b')
ax[0].set_ylabel("Power Output (kWh)")
ax[0].set_title("Hourly Power Output & Wave Displacement Over Time")
ax[0].legend()
ax[0].grid(True)

# 📊 Subplot 2: Wave Displacement Over Time
ax[1].plot(hourly_data_df['date'], hourly_data_df['wave_height'], label='Wave Displacement (m)', color='r')
ax[1].set_xlabel("Date & Time")
ax[1].set_ylabel("Wave Displacement (m)")
ax[1].legend()
ax[1].grid(True)

# 🕒 Auto-adjust x-axis formatting for date & time
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))  # Show Date + Time
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Show every 3 hours
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# Select a specific hour to visualize (e.g., first recorded hour)
selected_hour_index = 2  # Change this index to pick a different hour
selected_row = hourly_data_df.iloc[selected_hour_index]

# Extract time and wave displacement for that hour
wave_disp = selected_row['wave_displacement']

# Create time array for plotting (same length as wave_disp)
time_plot = np.arange(0, sim_time, dt)  # One-hour time scale

# Downsample factor (plot every nth point)
downsample_factor = 10  # Adjust this for more or less smoothing

# Select every nth point for cleaner visualization
time_plot_downsampled = time_plot[::downsample_factor]
wave_disp_downsampled = wave_disp[::downsample_factor]

# Plot Wave Displacement (Downsampled)
plt.figure(figsize=(12, 6))
plt.plot(time_plot_downsampled, wave_disp_downsampled, label="Wave Displacement (Smoothed)", color='r')

plt.xlabel("Time (s)")
plt.ylabel("Wave Displacement (m)")
plt.title("Wave Displacement Over One Hour (Downsampled)")
plt.legend()
plt.grid(True)
plt.show()
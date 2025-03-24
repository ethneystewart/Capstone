import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm
import random 

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
k = 500             # Spring constant (N/m)
B = 1               # Magnetic field strength (T)
N = 50              # Number of coil turns
c = 20              # Damping coefficient (Ns/m)

dt = 0.5            # Time step (s)
sim_time = 3600     # Simulation time per hour in seconds (1 hour)
time = np.arange(0, sim_time, dt)  # Define time array

# Global variables to store max and min forces over the entire period
global_max_forces = {'Morison': -np.inf, 'Spring': -np.inf, 'Damping': -np.inf, 'Total': -np.inf}
global_min_forces = {'Morison': np.inf, 'Spring': np.inf, 'Damping': np.inf, 'Total': np.inf}

# Load Excel file with data
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'Hourly Data'

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
    spectral density of ocean waves based on jonswap spectrum
    
    Hs: Significant wave height (m) - avg high of one-third of waves
    Tp: Peak wave period (s) - period at which max wave energy occurs 
    gamma: Peak enhancement factor (typically 3.3) - controls sharpness of the spectrum around the peak frequency

    returns spectral density S(f) at the given frequency f - which represents how wave energy  is distributed across different frequencies 
    """
    g = 9.81  # Gravity (m/s²)
    
    fp = 1 / Tp  # Peak frequency
    epsilon = 1e-6  # Small number to prevent division by zero
    f = np.maximum(f, epsilon)  # Ensure f is never zero
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

    return wave_disp, wave_vel, wave_acc, np.max(amplitudes)


def simulate_hourly_power(wave_height, wave_period):
    time = np.arange(0, sim_time, dt)
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)

    global global_max_forces, global_min_forces  # Track global min/max forces
    hourly_max_forces = {'Morison': -np.inf, 'Spring': -np.inf, 'Damping': -np.inf, 'Total': -np.inf}
    hourly_min_forces = {'Morison': np.inf, 'Spring': np.inf, 'Damping': np.inf, 'Total': np.inf}

    # Generate complex wave motion
    wave_disp, wave_vel, wave_acc, max_amplitude = complex_wave_displacement(time, wave_height, wave_period)
    
    for i in range(len(time) - 1):
        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel[i] - buoy_vel[i]) * abs(wave_vel[i] - buoy_vel[i]) +
                     Cm * V * rho * (wave_acc[i]))
        F_spring = k * buoy_disp[i]
        F_damp = c * buoy_vel[i]

        F_total = F_morison - F_spring - F_damp

        # Update min/max for each force component
        for force_name, force_value in zip(['Morison', 'Spring', 'Damping', 'Total'], 
                                           [F_morison, F_spring, F_damp, F_total]):
            hourly_max_forces[force_name] = max(hourly_max_forces[force_name], force_value)
            hourly_min_forces[force_name] = min(hourly_min_forces[force_name], force_value)

        
        buoy_acc[i + 1] = F_total / m
        buoy_vel[i + 1] = buoy_vel[i] + buoy_acc[i + 1] * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        V_out = -N * B * A_coil * buoy_vel[i + 1]
        Pout[i + 1] = V_out**2 / R

        # Update global min/max forces
    for force_name in global_max_forces.keys():
        global_max_forces[force_name] = max(global_max_forces[force_name], hourly_max_forces[force_name])
        global_min_forces[force_name] = min(global_min_forces[force_name], hourly_min_forces[force_name])

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

# Print total power and max/min forces
print(f"Total power over the entire period: {total_power} kWh")
print("\nMaximum Forces over the month:")
for force, value in global_max_forces.items():
    print(f"  {force}: {value:.2f} N")

print("\nMinimum Forces over the month:")
for force, value in global_min_forces.items():
    print(f"  {force}: {value:.2f} N")


plt.figure(figsize=(10, 5))
plt.plot(daily_summary_df['day'], daily_summary_df['total_power'], marker='o', linestyle='-')

# Format the x-axis labels
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format as "Dec 1"
plt.xticks(rotation=45)  # Rotate for readability

plt.xlabel('Date')
plt.ylabel('Total Power Output (kWh)')
plt.title('Total Power Output Over Time')
plt.grid()
plt.show()

#visualize the JONSWAP SPECTRAL DENSITY 
# Select an example row (e.g., first row)
example_row = df.iloc[random.randint(1, 32)]  # Change index for another row

# Extract example wave height (Hs) and wave period (Tp)
Hs_example = example_row['wave_height']
Tp_example = example_row['wave_period']
print(f"Example Data: Hs = {Hs_example}, Tp = {Tp_example}")

# Define frequency range for spectrum
frequencies = np.linspace(0, 0.5, 100)  # Adjust range as needed

# Compute JONSWAP spectrum values
spectrum_values = jonswap_spectrum(frequencies, Hs_example, Tp_example)

# Plot JONSWAP spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, spectrum_values, label=f'Hs={Hs_example}, Tp={Tp_example}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Density (m²/Hz)')
plt.title('JONSWAP Wave Spectrum for Extracted Data')
plt.legend()
plt.grid()
plt.show()











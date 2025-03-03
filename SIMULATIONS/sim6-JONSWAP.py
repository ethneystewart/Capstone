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

# Load Excel file with data
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'DEC1'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
except FileNotFoundError:
    raise FileNotFoundError(f"Excel file not found: {file_path}")
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the Excel file: {e}")

# Validate required columns - does it have them? 
required_columns = {'wave_height', 'wave_period', 'date'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing required columns in the dataset: {required_columns - set(df.columns)}")

# Convert 'date' column to datetime - so that we can see the hours easier 
df['date'] = pd.to_datetime(df['date'])

# Simulation parameters
depth_of_water = 27  # Depth of water (m)
k = 500              # Spring constant (N/m)
B = 1                # Magnetic field strength (T)
N = 50               # Number of coil turns
c = 20               # Damping coefficient (Ns/m)

dt = 0.1             # Time step (s)
sim_time = 3600      # Simulation time per hour in seconds (1 hour)

# --- JONSWAP Spectrum Function ---
def jonswap_spectrum(frequencies, Hs, Tp, gamma=3.3):
    """
    frequencies: array of wave frequencies (Hz) 
    Hs: signigicant wave height (m) - represents the avd of the highest 1/3 of waves 
    Tp: peak wave period (s) - the period of the dominant wave or most energetic wave 
    gamma: peak enhancement factor (default value is 3.3)
    outputs: array of spectral densities (m^2/Hz)
    """
    # Gravity (m/s^2)
    g = 9.81 
    # Peak frequency (Hz) - frequency with the most energy
    fp = 1 / Tp 
    # Empirical scaling factor - helps determine energy maginitude of spectrum 
    alpha = 0.076 * (Hs**2 * fp**4)  
    # Spectral width parameter - ensures smoother transition in wave energy distribution 
    sigma = np.where(frequencies <= fp, 0.07, 0.09)  
    # Peak enhancement factor - computes gaussian fct that shapes the spectrum ensure that energy is highest at fp and as it deviates from fp it smoothly decreases 
    r = np.exp(-((frequencies - fp) ** 2) / (2 * sigma**2 * fp**2))  
    # JONSWAP spectrum formula 
    S = alpha * (frequencies ** -5) * np.exp(-1.25 * (fp / frequencies) ** 4) * gamma ** r
    return S # returns spectrum valies for the coresponding wave frequencies 

# --- Generate JONSWAP Wave Time Series ---
def generate_wave_time_series(Hs, Tp, duration, dt, num_frequencies=100):
    """
    Hs: significant wave height (m) - represnet the average heigh of the hightest 1/3 of waves 
    Tp: peak wave period (s) - the period of the dominant wave or most energetic wave
    duration: total simulation time (s)
    dt: time step (s)
    num_frequencies: number of frequencies to use in wave construction 
    outputs: time array and wave elevation time series

    Generate a random wave elevation time series using the JONSWAP spectrum.
    Ensure that the generated wave height is consistent with Hs.
    """
    time = np.arange(0, duration, dt) # generates time array
    df = 1 / duration   # Frequency resolution - determines spacing between discrete frequencies 
    frequencies = np.linspace(0.01, 2 / Tp, num_frequencies)  # Frequency range - defines range of frequencies that contirbute to wave elevation
    spectrum = jonswap_spectrum(frequencies, Hs, Tp) # calls the jonswap_spectrum function to generate the spectrum values

    # Generate random phase angles - ensures that wave is realistic as waves in nature dont all start in sync 
    phases = np.random.uniform(0, 2 * np.pi, num_frequencies)

    # Construct wave time series via inverse FFT
    wave_elevation = np.zeros_like(time) # creates an empty array to store wave elevations 
    # loop through each frequency and calculate the wave elevation at each time step
    for i, f in enumerate(frequencies): 
        # np.sqrt(2 * spectrum[i] * df): Wave amplitude at this frequency
        # np.cos(2 * np.pi * f * time + phases[i]): Oscillatory wave motion
        # this sum creates a realistic wave pattern by combining multiple frequencies 
        wave_elevation += np.sqrt(2 * spectrum[i] * df) * np.cos(2 * np.pi * f * time + phases[i])

    # Ensure the generated wave has the correct significant wave height Hs
    generated_Hs = 4 * np.std(wave_elevation)  # Approximation of Hs - this is a mathematical equation from Rayleigh distribution 
    scaling_factor = Hs / generated_Hs
    wave_elevation *= scaling_factor  # Scale to match expected Hs - ensures we  simulating near our real world conditions and dont hav waves that are too weak or overloaded conditions 
    

    return time, wave_elevation

# --- Simulation Function ---
def simulate_hourly_power(wave_height, wave_period):
    '''
    wave_height: significant wave height (m) 
    wave_period: peak wave period (s)
    '''
    # creates wave elevation time series with given wave-height and wave period  - uses sim time of 3600s (1hr) with dt = 0.1s results in 36000 time steps
    time, wave_disp_series = generate_wave_time_series(wave_height, wave_period, sim_time, dt)

    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)


    for i in range(len(time) - 1):
        wave_disp = wave_disp_series[i]
        wave_vel = (wave_disp_series[i + 1] - wave_disp_series[i]) / dt
        wave_acc = (wave_vel - (wave_disp_series[i] - wave_disp_series[i - 1]) / dt) / dt if i > 0 else 0

        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel - buoy_vel[i]) * abs(wave_vel - buoy_vel[i]) +
                     Cm * rho * V * (wave_acc))
        F_total = F_morison - k * buoy_disp[i] - c * buoy_vel[i]

        buoy_vel[i + 1] = buoy_vel[i] + (F_total / m) * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        V_out = -N * B * A_coil * buoy_vel[i + 1]
        Pout[i + 1] = V_out**2 / R

    return time, buoy_disp, buoy_vel, Pout


# --- Run Simulation for Each Hour ---
hourly_data = []
daily_summary = []

# Store data for averaging
all_displacements = []
all_velocities = []
all_powers = []

for day, group in tqdm(df.groupby(df['date'].dt.date)):
    total_power_day = 0
    max_wave_height_day = 0

    for _, row in group.iterrows():
        wave_height = row['wave_height']
        wave_period = row['wave_period']
        time, buoy_disp, buoy_vel, Pout = simulate_hourly_power(wave_height, wave_period)

        hourly_data.append({
            'date': row['date'],
            'wave_height': wave_height,
            'wave_period': wave_period
        })

        # Store data for averaging
        all_displacements.append(buoy_disp)
        all_velocities.append(buoy_vel)
        all_powers.append(Pout)

# --- Compute Average Data for Plotting ---
if all_displacements:
    avg_displacement = np.mean(all_displacements, axis=0)
    avg_velocity = np.mean(all_velocities, axis=0)
    avg_power = np.mean(all_powers, axis=0)

# --- Plotting the Averaged Data ---
plt.figure(figsize=(12, 8))

# Downsample every 50th point for performance
downsample_factor = 50

plt.subplot(3, 1, 1)
plt.plot(time[::downsample_factor], avg_displacement[::downsample_factor], color='b', label='Avg Displacement')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Average Buoy Displacement over Time')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time[::downsample_factor], avg_velocity[::downsample_factor], color='g', label='Avg Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Average Buoy Velocity over Time')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time[::downsample_factor], avg_power[::downsample_factor], color='r', label='Avg Power Output')
plt.xlabel('Time (s)')
plt.ylabel('Power Output (W)')
plt.title('Average Power Output over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Clear memory
plt.close('all')

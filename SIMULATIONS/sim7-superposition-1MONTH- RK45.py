import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Define parameters for the simulation
rho = 1025          # Density of water (kg/m^3)
Cd = 0.6            # Drag coefficient
Cm = 1              # Added mass coefficient
A_buoy = 0.5        # Cross-sectional area of buoy (m^2)
V = 0.647944        # Displaced volume of buoy (m^3)
m = 609.06736       # Mass of buoy (kg)
A_coil = 0.1        # Coil area (m^2)
R = 72              # Electrical resistance (Ohms)

D = 0.342951*2 

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
sheet_name = 'Hourly Data'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
except FileNotFoundError:
    raise FileNotFoundError(f"Excel file not found: {file_path}")
except Exception as e:
    raise RuntimeError(f"An error occurred while reading the Excel file: {e}")

# Validate required columns
required_columns = {'wave_height', 'wave_period', 'date', 'ocean_current_velocity'}
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

def buoy_ode(t, y, wave_vel_func, wave_acc_func, current_velocity):
    disp, vel = y
    fluid_vel = wave_vel_func(t) + current_velocity
    rel_vel = fluid_vel - vel
    wave_acc = wave_acc_func(t)

    # Geometry and added mass area
    A_added = (np.pi / 4) * D**2

    # Effective mass (buoy mass + added mass)
    m_eff = m + Cm * A_added * rho

    # Morison force (now includes acceleration correctly)
    F_drag = 0.5 * rho * Cd * D * rel_vel * abs(rel_vel)
    F_added_mass = Cm * A_added * rho * wave_acc
    F_spring = k * disp
    F_damp = c * vel

    # Total force (reorganized)
    F_total = F_drag + F_added_mass - F_spring - F_damp

    # Final acceleration
    acc = F_total / m_eff

    return [vel, acc]


def simulate_hourly_power(wave_height, wave_period, current_velocity):
    time = np.arange(0, sim_time, dt)
    
    # Generate wave inputs
    wave_disp, wave_vel, wave_acc, max_amplitude = complex_wave_displacement(time, wave_height, wave_period)
    
    # Create interpolators for continuous input into solve_ivp
    wave_vel_func = interp1d(time, wave_vel, fill_value="extrapolate")
    wave_acc_func = interp1d(time, wave_acc, fill_value="extrapolate")
    
    # Initial state: [displacement, velocity]
    y0 = [0, 0]

    # Use RK45 integration (built into solve_ivp)
    sol = solve_ivp(buoy_ode, [0, sim_time], y0, t_eval=time, 
                    args=(wave_vel_func, wave_acc_func, current_velocity), method='RK45')

    buoy_disp = sol.y[0]
    buoy_vel = sol.y[1]

    # Compute electrical power
    V_out = -N * B * A_coil * buoy_vel
    Pout = V_out**2 / R

    total_power_hour = np.trapz(Pout, dx=dt) / 3600  # Convert from Ws to kWh
    
    return total_power_hour, max_amplitude



# Simulate for each hour and day
hourly_data = []
daily_summary = []

for day, group in tqdm(df.groupby(df['date'].dt.date)):
    total_power_day = 0
    max_wave_height_day = 0
    max_wave_period_day = 0  

    for _, row in group.iterrows():
        wave_height = row['wave_height']
        wave_period = row['wave_period']
        current_velocity = row['ocean_current_velocity']
        
        # Compute full wave displacement for this hour
        wave_disp, _, _, _ = complex_wave_displacement(time, wave_height, wave_period)
        
        power_hour, max_wave_height = simulate_hourly_power(wave_height, wave_period, current_velocity)

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
        max_wave_period_day = max(max_wave_period_day, row['wave_period'])
   
    daily_summary.append({
        'day': day,
        'total_power': total_power_day,
        'max_wave_height': max_wave_height_day
    })

# Convert results to DataFrames
hourly_data_df = pd.DataFrame(hourly_data)
daily_summary_df = pd.DataFrame(daily_summary)

hourly_df = pd.DataFrame(hourly_data)
# Add 'day' column for grouping
hourly_df['day'] = hourly_df['date'].dt.date

# Calculate daily averages
daily_avg_disp = hourly_df.groupby('day')['wave_displacement'].apply(lambda disps: np.mean([np.mean(d) for d in disps]))
daily_avg_period = hourly_df.groupby('day')['wave_period'].mean()

# Add to daily_summary_df
daily_summary_df['avg_wave_disp'] = daily_summary_df['day'].map(daily_avg_disp)
daily_summary_df['avg_wave_period'] = daily_summary_df['day'].map(daily_avg_period)

# Calculate total power over the entire period
total_power = daily_summary_df['total_power'].sum()

# Save results to Excel
with pd.ExcelWriter('wave_power_results.xlsx') as writer:
    hourly_data_df.to_excel(writer, sheet_name='Hourly Data', index=False)
    daily_summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)


print("\n=== Total Power Output Per Day ===")
for _, row in daily_summary_df.iterrows():
    print(f"{row['day']}: {row['total_power']:.3f} kWh")


print(f"Total power over the entire period: {total_power} kWh")

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# --- Plot 1: Total Power ---
axs[0].plot(daily_summary_df['day'], daily_summary_df['total_power'], 'b-o')
axs[0].set_ylabel('Total Power (kWh)')
axs[0].set_title('Daily Power Output')
axs[0].grid(True)

# --- Plot 2: Avg Wave Displacement ---
axs[1].plot(daily_summary_df['day'], daily_summary_df['avg_wave_disp'], 'g--s')
axs[1].set_ylabel('Avg Wave Disp (m)')
axs[1].set_title('Daily Average Wave Displacement')
axs[1].grid(True)

# --- Plot 3: Avg Wave Period ---
axs[2].plot(daily_summary_df['day'], daily_summary_df['avg_wave_period'], 'r--^')
axs[2].set_ylabel('Avg Wave Period (s)')
axs[2].set_title('Daily Average Wave Period')
axs[2].grid(True)

# Format x-axis only on bottom plot
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axs[2].set_xlabel('Date')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


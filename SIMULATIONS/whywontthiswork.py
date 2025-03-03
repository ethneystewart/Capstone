import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve



# Define parameters for the simulation
rho = 1025          # Density of water (kg/m^3)
Cd = 0.6            # Drag coefficient (no units)
Cm = 1              # Added mass coefficient (no units)
A_buoy = 0.5        # Cross-sectional area of buoy (m^2)
V = 1               # Displaced volume of buoy (m^3)
m = 768.5           # Mass of buoy (kg)
A_coil = 0.1        # Coil area (m^2)
R = 72              # Electrical resistance (Ohms)

# depth_of_water = 27  # Depth of water (m) - not used 
k = 500              # Spring constant (N/m)
B = 1                # Magnetic field strength (T)
N = 50               # Number of coil turns (integer)
c = 20               # Damping coefficient (Ns/m)

dt = 0.5            # Time step (s)
sim_time = 3600      # Simulation time per hour in seconds (1 hour)


# Load Excel file with data
file_path = 'DATA/DECEMBER 2024.xlsx'
sheet_name = 'DEC1'

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
wave_period = 7 
wave_height = 2.5
omega = 2 * np.pi / wave_period
amplitude = wave_height / 2



# wave_disp = amplitude * np.sin(omega * elapsed_time)
# wave_vel = amplitude * omega * np.cos(omega * elapsed_time)
# wave_acc = -amplitude * omega**2 * np.sin(omega * elapsed_time)

buoy_vel = 0.63
buoy_disp = 0.1

F_morison = (0.5 * rho * Cd * A_buoy * (0.63) * abs(0.63) +
                Cm  * V * (0.79))
print(F_morison)
F_spring = k * buoy_disp
F_damp = c * buoy_vel

F_total = F_morison - F_spring - F_damp
print(F_total)
buoy_acc = F_total / m

buoy_vel = buoy_vel + buoy_acc * dt
buoy_disp = buoy_disp + buoy_vel * dt

V_out = -N * B * A_coil * buoy_vel
Pout = V_out**2 / R

total_power_hour = np.sum(Pout) * dt / 3600

print(total_power_hour)






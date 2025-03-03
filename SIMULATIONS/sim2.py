import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define parameters for the simulation
rho = 1000          # Density of water (kg/m^3)
Cd = 0.6            # Drag coefficient
Cm = 1              # Added mass coefficient
A_buoy = 0.5        # Cross-sectional area of buoy (m^2)
V = 1               # Displaced volume of buoy (m^3)
m = 500             # Mass of buoy (kg)
A_coil = 0.1        # Coil area (m^2)
R = 10              # Electrical resistance (Ohms)

# for constant wave freq and amplitude 
wave_freq = 0.5     # Wave frequency (Hz)
wave_amp = 1        # Wave amplitude (m)

# define our time parameters 
sim_time = 100      # Simulation time (s)
dt = 0.01           # Time step (s)

# Define the parameters to vary
k_values = [500, 1000, 2000]  # Spring constant (N/m)
B_values = [1, 1.5, 2]        # Magnetic field strength (T)
N_values = [50, 100, 150]     # Number of coil turns
c_values = [20, 50, 100]      # Damping coefficient (Ns/m)

# Function to simulate the buoy dynamics
def simulate_buoy(rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt):

    # create an array containing evenly spaced values from 0 to our decided sim time(not including sim time) 
    # step in between two iterations is the dt 
    time = np.arange(0, sim_time, dt)

    # sets up arrays of all zeros same size as time 
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)

    #calculate our displacement, velocity and acceleration  - based off of a conisistant wave frequency and amplitude defined in our parameters 
    wave_disp = wave_amp * np.sin(2 * np.pi * wave_freq * time)
    wave_vel = 2 * np.pi * wave_freq * wave_amp * np.cos(2 * np.pi * wave_freq * time)
    wave_acc = -(2 * np.pi * wave_freq)**2 * wave_amp * np.sin(2 * np.pi * wave_freq * time)

    #iterate through each time step 
    for i in range(len(time) - 1):
        #calculate force interacting with the system
        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel[i] - buoy_vel[i]) * abs(wave_vel[i] - buoy_vel[i]) +
                     Cm * V * (wave_acc[i] - buoy_acc[i]))
        F_spring=k * buoy_disp[i] 
        F_damp = c * buoy_vel[i]

        #calculate total force 
        F_total = F_morison - F_spring - F_damp


        #Newtons Second law to determine our next time step acceleration 
        buoy_acc[i + 1] = F_total / m

        # euler integration  
        # velocity @ next time step = current velocity + small change in velocity due to next step acceleration
        # same procedure to find displacement  
        buoy_vel[i + 1] = buoy_vel[i] + buoy_acc[i + 1] * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        #Power Out  calculations dependent on the velocity of buoy 
        V_out = -N * B * A_coil * buoy_vel[i + 1]
        Pout[i + 1] = V_out**2 / R

    return time, buoy_disp, buoy_vel, Pout

# Initialize result dictionary 
results = []

# Iterate through all values we hope to vary 
for k in k_values:
    for B in B_values:
        for N in N_values:
            for c in c_values:
                time, buoy_disp, buoy_vel, Pout = simulate_buoy(
                    rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt
                )
                results.append({
                    "k": k, "B": B, "N": N, "c": c,
                    "time": time, "buoy_disp": buoy_disp,
                    "buoy_vel": buoy_vel, "Pout": Pout
                })

# finds maximum power output for each parameter change 
max_powers_k = []
for k in k_values:
    matching_results = [res for res in results if res["k"] == k]
    max_power = max([max(res["Pout"]) for res in matching_results])
    max_powers_k.append(max_power)

# plt.figure()
# plt.plot(k_values, max_powers_k, 'o-', linewidth=1.5)
# plt.grid(True)
# plt.xlabel('Spring Constant (k)')
# plt.ylabel('Max Power Output (W)')
# plt.title('Effect of Spring Constant on Max Power Output')
#plt.show()


# Repeat similar analysis for B and c
max_powers_B = []
for B in B_values:
    matching_results = [res for res in results if res["B"] == B]
    max_power = max([max(res["Pout"]) for res in matching_results])
    max_powers_B.append(max_power)

# plt.figure()
# plt.plot(B_values, max_powers_B, 'o-', linewidth=1.5)
# plt.grid(True)
# plt.xlabel('Magnetic Field Strength (B)')
# plt.ylabel('Max Power Output (W)')
# plt.title('Effect of Magnetic Field Strength on Max Power Output')
# #plt.show()

max_powers_c = []
for c in c_values:
    matching_results = [res for res in results if res["c"] == c]
    max_power = max([max(res["Pout"]) for res in matching_results])
    max_powers_c.append(max_power)

# plt.figure()
# plt.plot(c_values, max_powers_c, 'o-', linewidth=1.5)
# plt.grid(True)
# plt.xlabel('Damping Coefficient (c)')
# plt.ylabel('Max Power Output (W)')
# plt.title('Effect of Damping Coefficient on Max Power Output')

# all three graphs combined - PARAMETER VARYING 

plt.figure(figsize=(12, 10))

# Subplot 1: Max power output vs. spring constant (k)
plt.subplot(3, 1, 1)
plt.plot(k_values, max_powers_k, marker='o', label='Spring Constant (k)')
plt.xlabel('Spring Constant (k)')
plt.ylabel('Max Power Output (W)')
plt.title('Effect of Spring Constant on Max Power Output')
plt.grid(True)

# Subplot 2: Max power output vs. magnetic field strength (B)
plt.subplot(3, 1, 2)
plt.plot(B_values, max_powers_B, marker='s', label='Magnetic Field Strength (B)', color='orange')
plt.xlabel('Magnetic Field Strength (B)')
plt.ylabel('Max Power Output (W)')
plt.title('Effect of Magnetic Field Strength on Max Power Output')
plt.grid(True)

# Subplot 3: Max power output vs. damping coefficient (c)
plt.subplot(3, 1, 3)
plt.plot(c_values, max_powers_c, marker='^', label='Damping Coefficient (c)', color='green')
plt.xlabel('Damping Coefficient (c)')
plt.ylabel('Max Power Output (W)')
plt.title('Effect of Damping Coefficient on Max Power Output')
plt.grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# WHAT OCCURED DURING SIMULTATION
plt.figure(figsize=(10, 8))

# Subplot 1: Buoy Displacement
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
plt.plot(time, buoy_disp, label='Displacement', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Buoy Displacement')
plt.grid(True)

# Subplot 2: Buoy Velocity
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd plot
plt.plot(time, buoy_vel, label='Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Buoy Velocity')
plt.grid(True)

# Subplot 3: Power Output
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd plot
plt.plot(time, Pout, label='Power Output', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Power Output (W)')
plt.title('Power Output of Electromagnetic Harvester')
plt.grid(True)

# Adjust layout to prevent overlapping of subplots
plt.tight_layout()


# Find best combination of parameters
best_power = -np.inf    # initializes this as negative infinity 
best_params = {}

for res in results:
    max_power = max(res["Pout"])
    if max_power > best_power:
        best_power = max_power
        best_params = res

#what happens if we change our wave frequency how does that affect our power output 
wave_freq_values = np.linspace(0.1, 2.0, 20)  # Range of wave frequencies (Hz)
max_power_vs_freq = []

# Loop through different wave frequencies
for wave_freq in wave_freq_values:
    # Simulate buoy for a fixed parameter set (e.g., k, c, B, N)
    k = 1000  # Example spring constant
    c = 50    # Example damping coefficient
    B = 1.5   # Example magnetic field strength
    N = 100   # Example number of coil turns

    # Run simulation
    _, _, _, Pout = simulate_buoy(
        rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt
    )
    
    # Record the maximum power output for this frequency
    max_power_vs_freq.append(max(Pout))

# Plot the frequency response curve
plt.figure(figsize=(10, 6))
plt.plot(wave_freq_values, max_power_vs_freq, marker='o')
plt.xlabel('Wave Frequency (Hz)')
plt.ylabel('Max Power Output (W)')
plt.title('Frequency Response Curve')
plt.grid(True)
plt.tight_layout()
#plt.show()


#outputs best combination of the parameter arrays given 
print("Best Combination of Parameters:")
print(f"Spring Constant (k): {best_params['k']}")
print(f"Magnetic Field Strength (B): {best_params['B']}")
print(f"Number of Coil Turns (N): {best_params['N']}")
print(f"Damping Coefficient (c): {best_params['c']}")
print(f"Maximum Power Output: {best_power:.2f} W")


# Create a summary DataFrame for all simulations
simulation_summary = []

for res in results:
    max_Pout = max(res['Pout'])
    mean_Pout = np.mean(res['Pout'])
    std_Pout = np.std(res['Pout'])
    simulation_summary.append({
        'k': res['k'],
        'B': res['B'],
        'N': res['N'],
        'c': res['c'],
        'max_Pout': max_Pout,
        'mean_Pout': mean_Pout,
        'std_Pout': std_Pout
    })

simulation_summary_df = pd.DataFrame(simulation_summary)

# Create DataFrames for parameter variation data
parameter_k_df = pd.DataFrame({
    'k': k_values,
    'max_Pout': max_powers_k
})

parameter_B_df = pd.DataFrame({
    'B': B_values,
    'max_Pout': max_powers_B
})

parameter_c_df = pd.DataFrame({
    'c': c_values,
    'max_Pout': max_powers_c
})

# Create DataFrame for wave frequency data
wave_freq_df = pd.DataFrame({
    'wave_freq': wave_freq_values,
    'max_Pout': max_power_vs_freq
})

# Create DataFrame for best simulation data
best_simulation_df = pd.DataFrame({
    'time': best_params['time'],
    'buoy_disp': best_params['buoy_disp'],
    'buoy_vel': best_params['buoy_vel'],
    'Pout': best_params['Pout']
})

# Write all DataFrames to an Excel file with multiple sheets
with pd.ExcelWriter('simulation_results.xlsx') as writer:
    simulation_summary_df.to_excel(writer, sheet_name='Simulation Summary', index=False)
    parameter_k_df.to_excel(writer, sheet_name='Max Power vs k', index=False)
    parameter_B_df.to_excel(writer, sheet_name='Max Power vs B', index=False)
    parameter_c_df.to_excel(writer, sheet_name='Max Power vs c', index=False)
    wave_freq_df.to_excel(writer, sheet_name='Wave Frequency Response', index=False)
    best_simulation_df.to_excel(writer, sheet_name='Best Simulation Data', index=False)

plt.show()
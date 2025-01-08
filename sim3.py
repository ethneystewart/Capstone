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

wave_heights = np.linspace(1.75, 5.75, 20)
wave_periods = np.linspace(7, 15.1, 20)
depth_of_water = 10  # Example depth

def simulate_buoy(rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt, wave_heights, wave_periods, depth_of_water):
    def calculate_wave_length(T, d, tolerance=1e-6, max_iter=100):
        """Recursive wave length calculation."""
        g = 9.81  # Gravity
        L_prev = (g * T**2) / (2 * np.pi)  # Initial guess for L
        for _ in range(max_iter):
            L_next = (g * T**2) / (2 * np.pi) * np.tanh((2 * np.pi * d) / L_prev)
            if abs(L_next - L_prev) < tolerance:
                return L_next
            L_prev = L_next
        raise ValueError("Wave length did not converge")

    # Create an array containing evenly spaced values from 0 to our decided sim time (not including sim time)
    # Step in between two iterations is dt
    time = np.arange(0, sim_time, dt)

    # Setup arrays of all zeros, same size as time
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)

    # Simulate wave parameters
    wave_height = np.random.choice(wave_heights)  # Random wave height
    wave_period = np.random.choice(wave_periods)  # Random wave period

    try:
        wave_length = calculate_wave_length(wave_period, depth_of_water)
        wave_velocity = wave_length / wave_period
    except ValueError as e:
        print(f"Wave calculation error: {e}")
        return None, None, None, None

    # Calculate wave displacement, velocity, and acceleration
    wave_disp = wave_height * np.sin(2 * np.pi * wave_velocity * time / wave_length)
    wave_vel = (2 * np.pi * wave_velocity / wave_length) * wave_height * np.cos(2 * np.pi * wave_velocity * time / wave_length)
    wave_acc = -((2 * np.pi * wave_velocity / wave_length) ** 2) * wave_height * np.sin(2 * np.pi * wave_velocity * time / wave_length)

    # Iterate through each time step
    for i in range(len(time) - 1):
        # Calculate force interacting with the system
        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel[i] - buoy_vel[i]) * abs(wave_vel[i] - buoy_vel[i]) +
                     Cm * V * (wave_acc[i] - buoy_acc[i]))
        F_spring = k * buoy_disp[i]
        F_damp = c * buoy_vel[i]

        # Calculate total force
        F_total = F_morison - F_spring - F_damp

        # Newton's second law to determine the next time step acceleration
        buoy_acc[i + 1] = F_total / m

        # Euler integration
        buoy_vel[i + 1] = buoy_vel[i] + buoy_acc[i + 1] * dt
        buoy_disp[i + 1] = buoy_disp[i] + buoy_vel[i + 1] * dt

        # Power output calculations dependent on the velocity of the buoy
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
                    rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt,
                     wave_heights, wave_periods, depth_of_water
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
    best_simulation_df.to_excel(writer, sheet_name='Best Simulation Data', index=False)

plt.show()
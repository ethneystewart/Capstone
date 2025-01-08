#Ethney Stewart - 2024 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve


# Define parameters for the simulation
rho = 1025          # Density of water (kg/m^3) - correct - taylor
Cd = 0.6            # Drag coefficient
Cm = 1              # Added mass coefficient
A_buoy = 0.5        # Cross-sectional area of buoy (m^2)
V = 1               # Displaced volume of buoy (m^3)
m = 768.5           # Mass of buoy (kg) -correct - praveen
A_coil = 0.1        # Coil area (m^2)
R = 1000              # Electrical resistance (Ohms) - higher -praveen 

# for constant wave freq and amplitude 
wave_freq = 0.5     # Wave frequency (Hz)
wave_amp = 1        # Wave amplitude (m)

#want to come up with a random wave pattern
#EAST LONDON SOUTH AFRICA DATA
#this creates a array of num evenly spaced between our parameters 
# therefore an array of 20 different wave heights from  1.75- 5.75
wave_heights = np.linspace(1.75, 5.75, 50)
# same goes for periods 
wave_periods = np.linspace(7, 15.1, 100)
depth_of_water = 27  #meter

# define our time parameters 
sim_time = 200      # Simulation time (s)
dt = 0.01           # Time step (s)

# Define the parameters to vary
k_values = [500, 1000, 2000]  # Spring constant (N/m)
B_values = [1, 1.5, 2]        # Magnetic field strength (T)
N_values = [50, 100, 150]     # Number of coil turns
c_values = [20, 50, 100]      # Damping coefficient (Ns/m)


def simulate_buoy(rho, Cd, Cm, A_buoy, V, k, c, m, N, B, A_coil, R, wave_freq, wave_amp, sim_time, dt, wave_heights, wave_periods, depth_of_water):
    def calculate_wave_length(T, d):
        """Calculate wave length using a numerical solver."""
        g = 9.81  # Acceleration due to gravity

        # Define the function to solve
        def wave_eq(L):
            return L - (g * T**2) / (2 * np.pi) * np.tanh((2 * np.pi * d) / L)

        # Initial guess for wave length (deep water approximation)
        L_initial = (g * T**2) / (2 * np.pi)

        # Solve the equation
        L_solution, = fsolve(wave_eq, L_initial)

        return L_solution

    # Create an array containing evenly spaced values from 0 to our decided sim time (not including sim time)
    # Step in between two iterations is dt
    time = np.arange(0, sim_time, dt)

    # Setup arrays of all zeros, same size as time
    buoy_disp = np.zeros_like(time)
    buoy_vel = np.zeros_like(time)
    buoy_acc = np.zeros_like(time)
    Pout = np.zeros_like(time)
    wave_height_sim = np.zeros_like(time)
    wave_period_sim =np.zeros_like(time)
    amplitude_sim =np.zeros_like(time) 

    min_voltage = float('inf')
    max_voltage = float('-inf')
    min_disp = float('inf')
    max_disp = float('-inf')
    min_vel = float('inf')
    max_vel = float('-inf')
    min_acc = float('inf')
    max_acc = float('-inf')
    min_moris_force = float('inf')
    max_moris_force = float('-inf')
    min_spring_force =float('inf') 
    max_spring_force =float('-inf') 
    min_damp_force =float('inf') 
    max_damp_force =float('-inf') 
    min_total_force =float('inf') 
    max_total_force =float('-inf') 
    min_buoy_acc =float('inf') 
    max_buoy_acc =float('-inf') 
    min_buoy_vel =float('inf') 
    max_buoy_vel =float('-inf')
    min_buoy_disp =float('inf') 
    max_buoy_disp =float('-inf')
    min_power_out =float('inf') 
    max_power_out =float('-inf')

    # Initialize wave height and period
    wave_height = np.random.choice(wave_heights)
    wave_period = np.random.choice(wave_periods)

    # Define margins for variation
    height_margin = 0.5  # Max change in wave height (m)
    period_margin = 1.0  # Max change in wave period (s)
    # Iterate through each time step
    for i in range(len(time) - 1):
       
        # Update wave height and period dynamically every few seconds (e.g., every 5 seconds)
        if i % int(5 / dt) == 0:  # Update every 5 seconds
            # Adjust wave height and period within margins - smoother than if we just randomized the height and period again as it could jump weirldy 
            wave_height += np.random.uniform(-height_margin, height_margin) # generates random number in this margin and adds it too the current height 
            wave_height = np.clip(wave_height, wave_heights.min(), wave_heights.max())  # Keep within bounds - from our original def if it above or below bounds will put it back to boundary 

            wave_period += np.random.uniform(-period_margin, period_margin)
            wave_period = np.clip(wave_period, wave_periods.min(), wave_periods.max())  # Keep within bounds
            
            # Recalculate wave properties
            try:
                wave_length = calculate_wave_length(wave_period, depth_of_water)
            except ValueError as e:
                print(f"Wave calculation error during simulation: {e}")
                return None, None, None, None
            
            omega = 2 * np.pi / wave_period  # Recalculate angular frequency
            wave_number = 2 * np.pi / wave_length  # Recalculate wave number
            amplitude = wave_height / 2  # Recalculate amplitude
        wave_height_sim[i+1] = wave_height
        wave_period_sim [i+1] = wave_period
        amplitude_sim[i+1] = amplitude
        # Calculate wave displacement, velocity, and acceleration
        wave_disp = amplitude * np.sin(omega * time[i])  # Update displacement
        wave_vel = amplitude * omega * np.cos(omega * time[i])  # Update velocity
        wave_acc = -amplitude * omega**2 * np.sin(omega * time[i])  # Update acceleration

        # Calculate force interacting with the system
        F_morison = (0.5 * rho * Cd * A_buoy * (wave_vel - buoy_vel[i]) * abs(wave_vel - buoy_vel[i]) +
                    Cm * V * (wave_acc - buoy_acc[i]))
        F_spring = k * buoy_disp[i]
        F_damp = c * buoy_vel[i]
        #print(F_morison)

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
        min_voltage = min(min_voltage, V_out)
        max_voltage = max(max_voltage, V_out)
        min_disp = min(min_disp, wave_disp)
        max_disp = max(max_disp, wave_disp)
        min_vel = min(min_vel, wave_vel)
        max_vel = max(max_vel, wave_vel)
        min_acc = min(min_acc, wave_acc)
        max_acc = max(max_acc, wave_acc)
        min_moris_force = min(min_moris_force, F_morison)
        max_moris_force = max(max_moris_force, F_morison)
        min_spring_force = min(min_spring_force, F_spring)
        max_spring_force = max(max_spring_force, F_spring)
        min_damp_force = min(min_damp_force, F_damp)
        max_damp_force = max(max_damp_force, F_damp)
        min_total_force = min(min_total_force, F_total)
        max_total_force = max(max_total_force, F_total)
        min_buoy_disp = min(min_buoy_disp, buoy_disp[i + 1])
        max_buoy_disp = max(max_buoy_disp, buoy_disp[i + 1])
        min_buoy_vel = min(min_buoy_vel, buoy_vel[i + 1])
        max_buoy_vel = max(max_buoy_vel, buoy_vel[i + 1])
        min_buoy_acc = min(min_buoy_acc, buoy_acc[i + 1])
        max_buoy_acc = max(max_buoy_acc, buoy_acc[i + 1])
        min_power_out = min(min_power_out, Pout[i+1])
        max_power_out = max(max_power_out, Pout[i+1])
    return time, buoy_disp, buoy_vel, Pout , amplitude_sim , wave_height_sim, wave_period_sim, min_voltage, max_voltage, min_vel, max_vel, min_disp, max_disp, min_acc, max_acc, min_moris_force, max_moris_force, min_spring_force, max_spring_force, min_damp_force, max_damp_force, min_total_force, max_total_force, min_buoy_vel, max_buoy_vel, min_buoy_disp, max_buoy_disp, min_buoy_acc, max_buoy_acc, min_power_out, max_power_out


results = []

# Iterate through all values we hope to vary 
for k in k_values:
    for B in B_values:
        for N in N_values:
            for c in c_values:
                time, buoy_disp, buoy_vel, Pout, amplitude_data, height_data, period_data, min_voltage, max_voltage, min_vel, max_vel, min_disp, max_disp, min_acc, max_acc, min_moris_force, max_moris_force, min_spring_force, max_spring_force, min_damp_force, max_damp_force, min_total_force, max_total_force, min_buoy_vel, max_buoy_vel, min_buoy_disp, max_buoy_disp, min_buoy_acc, max_buoy_acc, min_power_out, max_power_out = simulate_buoy(
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

max_powers_B = []
for B in B_values:
    matching_results = [res for res in results if res["B"] == B]
    max_power = max([max(res["Pout"]) for res in matching_results])
    max_powers_B.append(max_power)

max_powers_c = []
for c in c_values:
    matching_results = [res for res in results if res["c"] == c]
    max_power = max([max(res["Pout"]) for res in matching_results])
    max_powers_c.append(max_power)

# all three graphs combined - PARAMETER VARYING 

plt.figure(figsize=(8, 8))


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

#simulate again with our best parameters to then plot our displacement, velocity, acceleration 
time, buoy_disp, buoy_vel, Pout, amplitude_data, height_data, period_data, min_voltage, max_voltage, min_vel, max_vel, min_disp, max_disp, min_acc, max_acc, min_moris_force, max_moris_force, min_spring_force, max_spring_force, min_damp_force, max_damp_force, min_total_force, max_total_force, min_buoy_vel, max_buoy_vel, min_buoy_disp, max_buoy_disp, min_buoy_acc, max_buoy_acc, min_power_out, max_power_out = simulate_buoy(
                    rho, Cd, Cm, A_buoy, V, best_params['k'], best_params['c'], m, best_params['N'], best_params['B'], A_coil, R, wave_freq, wave_amp, sim_time, dt,
                     wave_heights, wave_periods, depth_of_water)

# WHAT OCCURED DURING SIMULTATION
plt.figure(figsize=(8, 8))



# Subplot 1: Buoy Displacement With Best Parameters
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
plt.plot(time, buoy_disp, label='Displacement', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Best Buoy Displacement')
plt.grid(True)

# Subplot 2: Buoy Velocity With Best Parameters
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd plot
plt.plot(time, buoy_vel, label='Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Best Buoy Velocity')
plt.grid(True)

# Subplot 3: Power Output With Best Parameters
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd plot
plt.plot(time, Pout, label='Power Output', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Power Output (W)')
plt.title('Best Power Output of Electromagnetic Harvester')
plt.grid(True)

# Adjust layout to prevent overlapping of subplots
plt.tight_layout()

plt.figure(figsize=(8, 8))

# Subplot 1: Buoy Displacement With Best Parameters
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st plot
plt.plot(time, height_data, label='Wave Height', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Wave Height(m)')
plt.title('Random Wave Height Simulation')
plt.grid(True)

# Subplot 2: Buoy Velocity With Best Parameters
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd plot
plt.plot(time, period_data, label='Wave Period', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Wave Period(s)')
plt.title('Random Wave Period Simulation')
plt.grid(True)

# Subplot 3: Power Output With Best Parameters
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd plot
plt.plot(time, amplitude_data , label='Wave Amplitude', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Wave Amplitude')
plt.title('Wave Amplitude Simulation')
plt.grid(True)

# Adjust layout to prevent overlapping of subplots
plt.tight_layout()



#show all graphs 
plt.show()

data = {
    "Metric": [
        "Min Voltage (V)", "Max Voltage (V)",
        "Min Displacement (m)", "Max Displacement (m)",
        "Min Velocity (m/s)", "Max Velocity (m/s)",
        "Min Acceleration (m/s²)", "Max Acceleration (m/s²)",
        "Min Wave Force (N)", "Max Wave Force (N)",
        "Min Spring Force (N)", "Max Spring Force (N)",
        "Min Damping Force (N)", "Max Damping Force (N)",
        "Min Total Force (N)", "Max Total Force (N)",
        "Min Buoy Displacement (m)", "Max Buoy Displacement (m)",
        "Min Buoy Velocity (m/s)", "Max Buoy Velocity (m/s)",
        "Min Buoy Acceleration (m/s²)", "Max Buoy Acceleration (m/s²)",
        "Min Watts (W)", "Max Watts (W)"
    ],
    "Value": [
        min_voltage, max_voltage,
        min_disp, max_disp,
        min_vel, max_vel,
        min_acc, max_acc,
        min_moris_force, max_moris_force,
        min_spring_force, max_spring_force,
        min_damp_force, max_damp_force,
        min_total_force, max_total_force,
        min_buoy_vel, max_buoy_vel,
        min_buoy_disp, max_buoy_disp,
        min_buoy_acc, max_buoy_acc,
        min_power_out, max_power_out]
        
    }

# Create a DataFrame
df = pd.DataFrame(data)

# Export to Excel
with pd.ExcelWriter('simulation_results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Simulation Summary', index=False)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define constants
rho = 1000  # density of water (kg/m^3)
Cd = 1.2    # drag coefficient
Cm = 0.8    # added mass coefficient
A = 0.1     # cross-sectional area (m^2)
V = 0.05    # displaced volume (m^3)
k = 50      # spring constant (N/m)
c = 10      # damping coefficient (Ns/m)
m = 20      # mass of buoy (kg)
B = 1.0     # magnetic field strength (T)
N = 100     # number of turns of coil
R = 10      # resistance (ohms)

# Wave properties
omega = 2 * np.pi * 0.5  # wave angular frequency (rad/s)
amplitude = 1.0          # wave amplitude (m)

# Wave functions
def wave_velocity(t):
    return amplitude * omega * np.cos(omega * t)

def wave_acceleration(t):
    return -amplitude * omega**2 * np.sin(omega * t)

# Equation of motion
def equation_of_motion(t, y):
    x, x_dot = y
    v = wave_velocity(t)
    v_dot = wave_acceleration(t)
    # Morison force
    F_morison = (0.5 * rho * Cd * A * (v - x_dot) * abs(v - x_dot) +
                 Cm * V * (v_dot - x_dot))
    # Net force
    F_net = F_morison - k * x - c * x_dot
    x_ddot = F_net / m
    return [x_dot, x_ddot]

# Initial conditions
x0 = 0.0  # initial displacement (m)
x_dot0 = 0.0  # initial velocity (m/s)

# Time array
t_span = (0, 20)  # simulate for 20 seconds
t_eval = np.linspace(*t_span, 1000)  # evaluation points

# Solve the differential equation
solution = solve_ivp(equation_of_motion, t_span, [x0, x_dot0], t_eval=t_eval)

# Extract results
x = solution.y[0]
x_dot = solution.y[1]
t = solution.t

# Calculate power output
P_out = (-N * B * A * x_dot)**2 / R

# Plot results
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Displacement of Buoy")
plt.ylabel("x (m)")

plt.subplot(3, 1, 2)
plt.plot(t, x_dot)
plt.title("Velocity of Buoy")
plt.ylabel("x_dot (m/s)")

plt.subplot(3, 1, 3)
plt.plot(t, P_out)
plt.title("Power Output")
plt.ylabel("Power (W)")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

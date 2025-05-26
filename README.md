Capstone Project: Wave Energy Simulation

This was my final-year capstone project, completed in collaboration with three other students over an academic year. We aimed to design and simulate a wave energy harvesting system capable of converting ocean wave motion into electrical energy.

My primary contribution was leading the project's mathematical modelling and simulation aspect. I developed and implemented numerical simulations in Python to model the dynamic response of a buoy subjected to ocean wave forces. This involved:

Building simulations ranging from basic sinusoidal wave models to more complex representations using real-world data and the JONSWAP spectrum.

Integrating physical parameters such as buoyancy, damping, and wave excitation forces into differential equations that describe the system's motion.

Exploring different numerical integration methods, ultimately transitioning from Euler’s method to Runge-Kutta (RK45) for improved accuracy and stability.

Incorporating real wave data via the Open-Meteo Marine API to simulate realistic ocean conditions.

Collaborating closely with team members to interpret physical models, validate assumptions, and guide design decisions based on simulation results.

This project gave me hands-on experience translating physical systems into computational models, working with real environmental data, and applying advanced numerical techniques—all within a collaborative, multidisciplinary environment.

Dependencies
Install the following packages:

bash
Copy
Edit
pip install numpy matplotlib pandas tqdm openpyxl
pip install openmeteo-requests requests-cache retry-requests
Also requires:

python
Copy
Edit
from scipy.interpolate import interp1d  
from scipy.integrate import solve_ivp
Simulation Files Overview
sim1.py
Basic simulation using constant and sinusoidal wave forces.

sim2.py
Builds on sim1.py. Varies parameters (e.g., buoy mass, damping) and runs the simulation for each combination to find the optimal setup.

sim3.py
First attempt at using real wave data to calculate force more accurately. Currently has issues with converging wave height.

sim4.py
Used for midterm presentation. Randomizes wave height and period using min/max values and adjusts them every 5 seconds to simulate real ocean variability.

sim5.py
Integrates wave height API data into the simulation.

sim6.py (no superposition)
Uses the JONSWAP spectrum to simulate ocean waves with a single peak frequency. Simple sinusoidal approximation.

sim6.py with superposition
More advanced wave modeling using the JONSWAP spectrum + superposition to simulate realistic multi-frequency ocean waves.

sim6-JONSWAP.py
Does not run properly, but contains well-commented JONSWAP equations for reference.

sim7.py
Switches from Euler’s method to Runge-Kutta (RK45) for more accurate integration using SciPy.

Wave3D.py
Same as sim7.py but includes a 3D plot of the wave model at the end.

API Integration
GET_DATA.py
Fetches wave height, period, and direction data from the Open-Meteo Marine API.
Stores data in an Excel file with the following sheets:

DEC1 — Data from December 1st, 2024

DEC13 — Data from December 13th, 2024

Hourly Power — Hourly data for the full month of December 2024

Let me know if you'd like a version formatted for GitHub or with visuals/graphics included.







You said:
can you give me a version formatted for github


ChatGPT said:
Absolutely! Here's a GitHub-friendly version of your README, with clear headers, code blocks, and markdown formatting:

🌊 Capstone Project: Wave Energy Simulation
This project simulates a wave energy harvesting system using progressively more complex wave models, including real-world marine data and the JONSWAP spectrum.

📦 Requirements
Install dependencies using pip:

bash
Copy
Edit
pip install numpy matplotlib pandas tqdm openpyxl
pip install openmeteo-requests requests-cache retry-requests
Also used in later simulations:

python
Copy
Edit
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
📁 Simulation Files Overview
sim1.py
Basic simulation using a constant and sinusoidal wave force.

sim2.py
Builds on sim1.py.

Varies parameters (e.g., damping, mass) using arrays and runs simulations to find optimal configurations.

sim3.py
Attempts to calculate wave force using real wave height data.

Includes an iterative approach to determine wave force, but currently has convergence issues.

sim4.py
Midterm version.

Uses randomized wave height and period (based on min/max ranges) and adjusts them every 5 seconds to mimic natural ocean changes.

sim5.py
First integration of wave height API data into the simulation.

sim6.py (no superposition)
Uses the JONSWAP spectrum to simulate ocean waves using only the dominant frequency.

Simple sinusoidal wave approximation.

sim6.py with superposition
Advanced modeling using the JONSWAP spectrum + superposition:

Wave energy is distributed across multiple frequencies.

Components are summed with randomized phases for realistic ocean behavior.

sim6-JONSWAP.py
This version does not work properly, but contains well-commented JONSWAP equations for educational/reference purposes.

sim7.py
Replaces Euler's method with RK45 (Runge-Kutta) integration via solve_ivp from SciPy.

More accurate and stable for dynamic wave systems.

Wave3D.py
Same as sim7.py, but includes a 3D plot visualizing the final wave model output.

🌐 API Integration
GET_DATA.py
Pulls hourly and daily marine weather data from the Open-Meteo Marine API.

Data includes:

Wave height

Wave period

Wave direction

📊 Data is saved into an Excel file with the following sheets:

DEC1 — December 1st, 2024

DEC13 — December 13th, 2024

Hourly Power — All hourly data from December 2024

📌 Notes
Some simulations are exploratory and experimental — see comments in files for more context.

The JONSWAP model offers both a simplified and realistic version of wave formation. Choose based on the fidelity needed.

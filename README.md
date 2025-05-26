#  Capstone Project: Wave Energy Simulation

This was my final-year capstone project, completed in collaboration with three other students over an academic year. We aimed to design and simulate a wave energy harvesting system capable of converting ocean wave motion into electrical energy.

## My Role

I led the **mathematical modelling and simulation** component of the project. I developed and implemented numerical simulations in Python to model the dynamic response of a buoy subjected to ocean wave forces.

This involved:

- Building simulations ranging from basic sinusoidal wave models to more complex representations using real-world data and the JONSWAP spectrum.
- Integrating physical parameters such as buoyancy, damping, and wave excitation forces into differential equations describing system motion.
- Exploring different numerical integration methods, ultimately transitioning from **Euler’s method to Runge-Kutta (RK45)** for improved accuracy and stability.
- Incorporating real marine data from the **Open-Meteo Marine API** to simulate realistic ocean conditions.
- Collaborating closely with the team to interpret physical models, validate assumptions, and guide design decisions based on simulation results.

This project gave me hands-on experience in computational modelling of physical systems, handling environmental datasets, and applying advanced numerical techniques within a multidisciplinary team.

---

## Dependencies

Install required packages:

```bash
pip install numpy matplotlib pandas tqdm openpyxl
pip install openmeteo-requests requests-cache retry-requests
pip install scipy
```
## 📁 Simulation Files Overview

### `sim1.py`
- Basic simulation using **constant** and **sinusoidal** wave forces.

### `sim2.py`
- Expands on `sim1.py`.
- Varies parameters (e.g., mass, damping) and runs multiple simulations to find the optimal configuration.

### `sim3.py`
- First attempt to use **real wave height data** to calculate wave forces.
- Currently experiences convergence issues in wave height calculations.

### `sim4.py`
- **Midterm version**.
- Randomizes wave height and period values (within a range) every 5 seconds to simulate natural wave variability.

### `sim5.py`
- Begins integration of **live wave data from an API** into the simulation.

### `sim6.py` *(no superposition)*
- Uses the **JONSWAP spectrum** to model waves using only the **dominant frequency** (simplified approach).

### `sim6.py with superposition`
- Implements a **realistic multi-frequency wave model**:
  - Calculates energy distribution using JONSWAP.
  - Applies superposition of frequency components with randomized phases to simulate complex ocean wave behavior.

### `sim6-JONSWAP.py`
- Not fully functional.
- Includes **well-commented** JONSWAP equations for reference and understanding.

### `sim7.py`
- Replaces Euler’s method with **RK45 integration** (`solve_ivp`) for improved numerical accuracy.

### `Wave3D.py`
- Version of `sim7.py` that outputs a **3D visualization** of wave motion.

---

## API Integration

### `GET_DATA.py`

Fetches marine weather data using the [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api), including:

- Wave height
- Wave period
- Wave direction

Data is saved into an Excel file with the following sheets:

- `DEC1` — Data from December 1st, 2024  
- `DEC13` — Data from December 13th, 2024  
- `Hourly Power` — Hourly data for the full month of December 2024

---

## Notes

- Some simulation files are exploratory or iterative; refer to comments within the scripts for development details.
- The **JONSWAP model** is used in two versions: a simplified peak-frequency model and a more complex multi-frequency version with superposition.
- Wave realism increases progressively from `sim1.py` through `sim6.py`, culminating in a fully dynamic RK45 simulation in `sim7.py`.



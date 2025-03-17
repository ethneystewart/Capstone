# Capstone

Requirements: 
numpy 
matplotlib.pyplot 
pandas
tqdm

sim1.py - starting chatgpt simulation build simple sinusodial and constant wave force

sim2.py - still uses a constant wave force and simple sinusodial varies parameters using array and then runs simulation through the varying params to find best choice 

sim3.py - first attempt at making it more complicated utilizing wave height and new data to make wave force more accurate currently errors trying to converge wave height calculation

sim4.py - midterm presentation simulation - uses wave height and wave period data from study and then randomizes wave height and period by choosing a value between min and max possible wave hight and period. It also varies the wave height and period within a certain margin every 5 seconds to simulate the changing wave. 

Requirements: 

pip install openmeteo-requests
pip install requests-cache retry-requests numpy pandas

api website: https://open-meteo.com/en/docs/marine-weather-api#latitude=-33.0153&longitude=27.9116&current=wave_height,wave_direction,wave_period&hourly=wave_height,wave_direction,wave_period,wind_wave_height,wind_wave_direction,wind_wave_period,wind_wave_peak_period,swell_wave_height,swell_wave_direction,swell_wave_period,swell_wave_peak_period,ocean_current_velocity,ocean_current_direction&daily=wave_height_max,wave_direction_dominant,wave_period_max&timezone=auto&start_date=2024-12-01&end_date=2024-12-31&time_mode=time_interval

GET_DATA.py - this get data from an api for hourly and daily maxes of wave height period and direction around east london and stores it in a excel file 

sim5.py - attempt to integrate a waveheight api into our simulation 

sim6.py with superposition  - integrates a more complex wave - The JONSWAP spectrum defines how wave energy is distributed across different frequencies by calculating spectral density values (S_f). This determines the amplitude of each frequency component. The superposition method then takes these amplitude-scaled frequency components and sums them with randomized phases, reconstructing a realistic multi-frequency ocean wave. Essentially, JONSWAP sets the energy distribution, and superposition assembles the final wave motion.

sim6.py - no superposiiton - just integrates JONSWAP - The JONSWAP spectrum in this code calculates the wave energy density at the peak frequency only, simplifying the wave model. The superposition method is not used here; instead, the wave motion is simulated using just the dominant frequency component, assuming a sinusoidal motion to approximate real ocean waves.
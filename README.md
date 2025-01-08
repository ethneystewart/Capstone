# Capstone

Requirements: 
numpy 
matplotlib.pyplot 
pandas
scipy.optimize

sim1.py - starting chatgpt simulation build simple sinusodial and constant wave force

sim2.py - still uses a constant wave force and simple sinusodial varies parameters using array and then runs simulation through the varying params to find best choice 

sim3.py - first attempt at making it more complicated utilizing wave height and new data to make wave force more accurate currently errors trying to converge wave height calculation

sim4.py - midterm presentation simulation - uses wave height and wave period data from study and then randomizes wave height and period by choosing a value between min and max possible wave hight and period. It also varies the wave height and period within a certain margin every 5 seconds to simulate the changing wave. 

Requirements: 

pip install openmeteo-requests
pip install requests-cache retry-requests numpy pandas

sim5.py - attempt to integrate a waveheight api into our simulation 
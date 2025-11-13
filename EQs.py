import math
import numpy as np
import pickle


# This realies on all my arrays being the same length
AOA_list = [-3 + 0.5*i for i in range(31)] # Angle of Attack in degrees
CL_list = [.0806*a+.2495 for a in AOA_list] # Lift Coefficient
CD_list = [.0271+.0534*x**2 for x in CL_list] # Drag oefficient

#Model Constants
V = 110 # Velocity in ft/s
RHO = 0.00242 # Air Density in slugs/ft^3
S = 0.77 # Wing Area in ft^2
W = 10 # Weight in lbs

# Initializing array to store results
N_mean_list = np.zeros(len(CL_list))
A_mean_list = np.zeros(len(CL_list))
data = []

#Creates a range to run through all values of input arrays
for i in range(len(CL_list)):
    
    # Getting current values from input arrays
    CL = CL_list[i] 
    CD = CD_list[i]
    AOA = AOA_list[i]
    
    #Converts degrees to radians
    A = float(np.radians(AOA)) # Angle of Attack in radians

    # Equations to get L and D
    L = 0.5 * RHO * V**2 * S * CL # Lift in lbs
    D = 0.5 * RHO * V**2 * S * CD # Drag in lbs

    # Equations to get N_aero and A_aero
    N_aero = L * math.cos(A) + D * math.sin(A) # Aerodynamic Normal Force in lbs
    A_aero = - L * math.sin(A) + D * math.cos(A) # Aerodynamic Axial Force in lbs

    # Equations to get N_meas and A_meas
    N_meas = N_aero + W * math.sin(A) # Measured Normal Force in lbs
    A_meas = A_aero + W * math.cos(A) # Measured Axial Force in lbs

    #Stores all values as data
    data.append({'NF': N_meas, 'AF': A_meas, 'SF': 0, 'PM': 0, 'RM': 0,'YM': 0, 'Theta': AOA, 'Q': 0.5 * RHO * V**2, 'S': S, 'W': W})

testinfo = {'S': S, 'W': W, 'RunNum': 1}

# Wind tunnel speed and angles (AOA, model constants, dynamic pressure (q), desity sea level)
#saves the dictioary as a pickle file 
with open("wind_forces_inputs.pkl", "wb") as f: 
    pickle.dump((data, testinfo), f)

#prints the results to check accuracy
print('end')


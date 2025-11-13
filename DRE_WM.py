import math
import numpy as np
import pickle

'''OLD CODE INCASE I HAVE TO GO BACK# Values from testing
N_array = np.array([58.89963282, 58.76497715, 58.89963282]) #Normal Force in lb 
A_array = np.array([13.53283359, 13.55651248, 13.53283359]) #Axial Force in lb
AOA_array = np.array([5.74,5.72,5.74]) # Angle of Attack in degrees

#Temperary figure out how to calculate from constants you may have
PitchingMoment_array = np.zeros_like(N_array) # Pitching Moment in lb*ft
PitchingMoment_array[:] = 1  #Example constant value [change as needed]

# Zero arrays, not accounted for currently
SideForce_array = np.zeros_like(N_array) # Side Force in lb
RollMoment_array = np.zeros_like(N_array) # Roll Moment in lb*ft
YawMoment_array = np.zeros_like(N_array)# Yaw Moment in lb*ft

#Model Constants
V = 95.33 # Velocity in ft/s
RHO = 0.002378 # Air Density in slugs/ft^3
S = 7.44 # Wing Area in ft^2
W = 15 # Weight in lbs

# Initializing array to store results
CL_array = np.zeros_like(N_array)   # Lift Coefficient
CD_array = np.zeros_like(N_array)   # Drag Coefficient
data = []


#Creates a range to run through all values of input arrays
for i in range(len(A_array)):
    N_meas = N_array[i]
    A_meas = A_array[i]

    #Converts degrees to radians
    A = math.radians(AOA_array[i]) # Angle of Attack in radians

    # Equations to get N_meas and A_meas
    N_aero = N_meas - W * math.sin(A) # Aerodynamic Normal Force in lbs
    A_aero = A_meas - W * math.cos(A) # Aerodynamic Axial Force in lbs

    # Equations to get N_aero and A_aero
    L = N_aero * math.cos(A) - A_aero * math.sin(A) # Lift in lbs
    D = N_aero * math.sin(A) + A_aero * math.cos(A) # Drag in lbs

    # Calculate coefficients
    C_L = L / (0.5 * RHO * V**2 * S) # Lift Coefficient
    C_D = D / (0.5 * RHO * V**2 * S) # Drag Coefficient

    # Store values back into arrays
    CL_array[i] = C_L
    CD_array[i] = C_D

    AOA = A
    #Stores all values as data
    data.append({'NF': N_meas, 'AF': A_meas, 'SF': 0, 'PM': 0, 'RM': 0,'YM': 0, 'Theta': AOA, 'Q': 0.5 * RHO * V**2, 'S': S, 'W': W})


# Wind tunnel speed and angles (AOA, model constants, dynamic pressure (q), desity sea level)

#saves the dictioary as a pickle file
with open("wind_forces_results.pkl", "wb") as f: 
    pickle.dump(data, f)

print (CD_array, CL_array)'''


def DREs(input_data, G=None):
    """Minimal wrapper to compute CL/CD from a list of input dicts.

    Keeps the same calculation form used above. Expects input_data to be
    a list of dicts with keys 'NF','AF','Theta'. Returns a list of dicts
    including 'CL' and 'CD'.
    """
    out = []

    # Minimal local defaults (used if G is not provided). These mirror
    # the constants used previously in the module's example code.
    V_default = 95.33
    RHO_default = 0.002378
    S_default = 7.44
    W_default = 15

    # allow G to override S and W, otherwise use defaults
    S_local = G.get('S') if isinstance(G, dict) and 'S' in G else S_default
    W_local = G.get('W') if isinstance(G, dict) and 'W' in G else W_default
    #Q = 0.5 * RHO_default * V_default**2

    for e in input_data:
        N_meas = e.get('NF', 0)
        A_meas = e.get('AF', 0)
        Theta = e.get('Theta', 0)
        Q = e.get('Q', 0)
        A_rad = math.radians(Theta)

        N_aero = N_meas - W_local * math.sin(A_rad)
        A_aero = A_meas - W_local * math.cos(A_rad)

        L = N_aero * math.cos(A_rad) - A_aero * math.sin(A_rad)
        D = N_aero * math.sin(A_rad) + A_aero * math.cos(A_rad)

        C_L = L / (Q * S_local)
        C_D = D / (Q * S_local)

        out.append({
            'NF': N_meas,
            'AF': A_meas,
            'SF': e.get('SF', 0),
            'PM': e.get('PM', 0),
            'RM': e.get('RM', 0),
            'YM': e.get('YM', 0),
            'Theta': Theta,
            'Q': Q,
            'S': S_local,
            'W': W_local,
            'CL': C_L,
            'CD': C_D,
        })

    return out






import os
import numpy as np
import matplotlib.pyplot as plt
import WINDMONTE_utils
from scipy import stats
import pickle
from tqdm import tqdm
import time
from scipy.stats import norm


""" 
WINDMONTE.py is a "WIND" tunnel "MONTE" Carlo simulation for uncertainty propagation and sensitivity analysis of wind tunnel test data.  

Author: Drew Curriston
Date: 28 February, 2024
Location: Oran W. Nicks Low Speed Wind Tunnel, Texas A&M University

The original code was developed independently and toward fulfillment of doctoral program requirements as published in "WIND TUNNEL DATA 
QUALITY ASSESSMENT AND IMPROVEMENT THROUGH INTEGRATION OF UNCERTAINTY ANALYSIS IN TEST DESIGN," PhD Dissertation, Texas A&M University, 2024.

This version is a generic copy that removes all code specific to data I/O at the Oran W. Nicks LSWT, removes code relating to the analysis 
of multiple objectives covered in the dissertation, provides additional in-line comments, and removes MCM convergence criteria in an effort 
to simplify.  This version of the code limits some of the capabilities that were used in completion of the dissertation in an attempt to improve 
utility. 

The code is separated into three regions, which are commented within the code: 1) Inputs, 2) MCM simulation, and 3) Plotting results.

TO-DO LIST:
Add planning predictions function from aero coefficients
Add NRT random uncertainty option
Add replicate data random uncertainty contributions

"""
tic = time.time()

#region 1: Data input and MCM setup (Check to ensure all data, parameters, and error sources are correct for analysis)

# 1.1 Input MCM parameters
M = 1000  # Number of MCM trials to run.  
outputfile = 'WINDMONTE_outputs.pkl'  # specify output file
UPCs = True  # Set to True to compute the UPCs
UPC_M = 100  # Set the number of trials to simulate for each error source in calculating UPCs

# 1.2 Load test data

""" 
Read test data from .pkl file.  Replace this section with code that reads in either data from a file or input from data acquisition system in the same format.

data: list of dictionaries for One-Factor-At-a-Time (OFAT) test run.  Each item in list is a data point, each data point has a dictionary of measurands and constants as inputs.
testinfo: test constants, notes, information pertaining to all data points in test run
data_multisample: Same as data, but in arrays of multisample data rather than a single float value.  Used to calculate random uncertainty for near-real-time analysis.

This version has 3 options to load data:  
    1.  Load_source = 'LSWT_raw':  Load raw data files in the format used at the LSWT, only applicable at the LSWT
    2.  Load_source = '*.pkl':  Load a .pkl file with variables "data", "testinfo", and "data_multisample" using the formats specified above.
    3.  Load_source = '*.csv':  Load a .csv file that has the "data" and "testinfo" variables written in the format of 'inputdata_example.csv' provided.  This option does not have multisample data for propagating random uncertainty from that source.
"""

Load_source = 'inputdata_example.pkl' #'LSWT_raw' #'inputdata_example.csv'  

# Uncomment to load data from .pkl file
""" Load_source = 'inputdata_example.pkl'
with open(Load_source, 'rb') as f:  
    data,data_multisample,testinfo = pickle.load(f) """

# Uncomment to load data from .csv file
Load_source = 'inputdata_example.csv'
data,testinfo = WINDMONTE_utils.load_csv_data(Load_source)

# Add section to generate predicted data for test planning

WINDMONTE_utils.check_list_of_dicts(data)  # utility to check and make sure input data is in the right format

# 1.3 Define systematic uncertainty for elemental error sources 
U_systematic = WINDMONTE_utils.U_systematic() # instantiate from systematic uncertainty class

# add systematic elemental error sources (see WINDMONTE_README.doc)
U_systematic.add_error_source(measurements=['Theta'],distribution='norm',params=[0,0.0045],source='b_Q-flex',units='deg')
U_systematic.add_error_source(measurements=['Psi'],distribution='norm',params=[0,0.0295],source='b_Psi',units='deg')
U_systematic.add_error_source(measurements=['Phi'],distribution='norm',params=[0,0.025],source='b_Phi',units='deg')
U_systematic.add_error_source(measurements=['Qset','Qact'],distribution='norm',params=[0,0.119],source='b_Qcal',units='psf')
U_systematic.add_error_source(measurements=['Pstat'],distribution='norm',params=[0,0.005],source='b_P_stat',units='psf')
U_systematic.add_error_source(measurements=['Ptot'],distribution='norm',params=[0,0.005],source='b_P_tot',units='psf')
U_systematic.add_error_source(measurements=['Baro'],distribution='norm',params=[0,0.025],source='b_P_baro',units='psf')
U_systematic.add_error_source(measurements=['Temp'],distribution='norm',params=[0,0.05],source='b_T',units='deg F')
U_systematic.add_error_source(measurements=['TempT'],distribution='norm',params=[0,0.05],source='b_T0',units='deg F')
U_systematic.add_error_source(measurements=['NF'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_NF',units='lbf')  # if percent_scale set to True, will scale by nominal value and param #2 is (sigma/nominal value)
U_systematic.add_error_source(measurements=['SF'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_SF',units='lbf')
U_systematic.add_error_source(measurements=['AF'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_AF',units='lbf')
U_systematic.add_error_source(measurements=['PM'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_PM',units='in.lbf')
U_systematic.add_error_source(measurements=['RM'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_RM',units='in.lbf')
U_systematic.add_error_source(measurements=['YM'],distribution='norm',params=[0,0.0025],percent_scale=True,source='b_YM',units='in.lbf')


# 1.4 Define random uncertainty for Variables of Interest (VOIs) using direct comparison of replicate data --OR-- define random uncertainty for elemental error sources 
U_random = WINDMONTE_utils.U_random() # instantiate from random uncertainty class

s_flag = 'P'  # Choose methodology for random uncertainty: 
    # "P" to propagate from variable U_random, 
    # "DCR" to determine VOI uncertainty from direct comparison of replicate data.  Must have replicate data and a function defined to populate that data.

if s_flag == 'P':
    # add random elemental error sources to propagate with MCM
    U_random.add_error_source(measurements=['Theta'],distribution='norm',params=[0,0.0088],source='s_Q-flex',units='deg')
    U_random.add_error_source(measurements=['Psi'],distribution='norm',params=[0,0.005],source='s_Psi',units='deg')
    U_random.add_error_source(measurements=['Phi'],distribution='norm',params=[0,0.005],source='s_Phi',units='deg')
    U_random.add_error_source(measurements=['Qset','Qact'],distribution='norm',params=[0,0.06],source='s_Q',units='psf')
    U_random.add_error_source(measurements=['Pstat'],distribution='norm',params=[0,0.016],source='s_P_stat',units='psf')
    U_random.add_error_source(measurements=['Ptot'],distribution='norm',params=[0,0.013],source='s_P_tot',units='psf')
    U_random.add_error_source(measurements=['Baro'],distribution='norm',params=[0,0.06],source='s_P_baro',units='psf')
    U_random.add_error_source(measurements=['Temp'],distribution='norm',params=[0,0.09],source='s_T',units='deg F')
    U_random.add_error_source(measurements=['TempT'],distribution='norm',params=[0,0.09],source='s_T0',units='deg F')
    U_random.add_error_source(measurements=['NF'],distribution='norm',params=[0,0.077],source='s_NF',units='lbf')  # if percent_scale set to True, will scale by nominal value and param #2 is (sigma/nominal value)
    U_random.add_error_source(measurements=['SF'],distribution='norm',params=[0,0.031],source='s_SF',units='lbf')
    U_random.add_error_source(measurements=['AF'],distribution='norm',params=[0,0.038],source='s_AF',units='lbf')
    U_random.add_error_source(measurements=['PM'],distribution='norm',params=[0,0.026],source='s_PM',units='in.lbf')
    U_random.add_error_source(measurements=['RM'],distribution='norm',params=[0,0.046],source='s_RM',units='in.lbf')
    U_random.add_error_source(measurements=['YM'],distribution='norm',params=[0,0.010],source='s_YM',units='in.lbf')
elif s_flag == 'DCR':
    # if direct comparison of replicate data is used, 
    pass  # to be coded

#endregion


#region 2: Run MCM Simulation for propagation of systematic uncertainty
RunData = WINDMONTE_utils.MCM_sim(data, testinfo, U_systematic, U_random, s_flag, M)

toc = time.time()
print('Total time elapsed = ' + str(toc-tic))
print('Run ' + str(testinfo['RunNum']) + ' complete')

# Calculate UPCs if desired 
if UPCs:
    WINDMONTE_utils.UPCs(RunData, data, testinfo, U_systematic, U_random, s_flag, UPC_M)
    
# Save results
with open(outputfile, 'wb') as f:  
    pickle.dump([RunData,U_systematic,U_random],f)
print("Simulation data saved to file: {}".format(outputfile))


#region 3: Plot results or run the WINDMONTE_GUI code
os.system("python WINDMONTE_GUI.py")

# or plot directly using plot behaviors for the TestRun, DataPoint, or VOI variable class
""" Plot_VOIs = ['CD', 'CY','CL','Cl','Cm','Cn']
RunData.plot_errorbars('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',['L_D','L12_D','L32_D'],ncols=3) 
plt.show()"""
#endregion

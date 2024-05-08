# WINDMONTE is licensed under GNU GPL v3, see COPYING.txt

import os
import numpy as np
import matplotlib.pyplot as plt
import utilities
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
M = 1000  # Number of MCM trials to run.  Recommend >500 
outputfile = 'output_data.pkl'  # specify output file
UPCs = True  # Set to True to compute the UPCs
UPC_M = 100  # Set the number of trials to simulate for each error source in calculating UPCs.  Recommend >100.

s_flag = 'DCR'  # Choose methodology for random uncertainty: 
    # "P" to propagate from variable U_random, 
    # "DCR" to determine VOI uncertainty from direct comparison of replicate data.  Must have replicate data and a function defined to populate that data.
    # "None" to only evaluate systematic uncertainty

# 1.2 Load test data

""" 
Read test data from .pkl file.  Replace this section with code that reads in either data from a file or input from data acquisition system in the same format.

data: list of dictionaries for One-Factor-At-a-Time (OFAT) test run.  Each item in list is a data point, each data point has a dictionary of measurands and constants as inputs.
testinfo: test constants, notes, information pertaining to all data points in test run
data_multisample: Same as data, but in arrays of multisample data rather than a single float value.  Used to calculate random uncertainty for near-real-time analysis.

This version has 3 options to load data:  
    1.  Load_source = '*.pkl':  Load a .pkl file with variables "data", "testinfo", and "data_multisample" using the formats specified above.
    2.  Load_source = '*.csv':  Load a .csv file that has the "data" and "testinfo" variables written in the format of 'inputdata_example.csv' provided.  This option does not have multisample data for propagating random uncertainty from that source.
    3.  Add function to generate predicted test data based on test planning estimates.
"""

# Uncomment to load data from .pkl file
Load_source = 'inputdata_example.pkl'
with open(Load_source, 'rb') as f:  
    data,data_multisample,testinfo = pickle.load(f)

# Uncomment to load data from .csv file
""" Load_source = 'inputdata_example.csv'
data,testinfo = utilities.load_csv_data(Load_source) """

# If using predicted data for test planning, input data here

utilities.check_list_of_dicts(data)  # utility to check and make sure input data is in the right format

# 1.3 Define systematic uncertainty for elemental error sources 
U_systematic = utilities.U_systematic() # instantiate from systematic uncertainty class

""" Add systematic elemental error sources (see WINDMONTE_README.doc)
follow the format used in the example:
    - measurements: defines which measurements (keys in data[i] dictionaries) this error source affects
    - distribution/params: the type of PDF and parameters as defined in the scipy.stats documentation.  
    - source: label the source for plotting later
    - units: list the units, must match the units in the 'data' variable nominal values for each measurement the error source applies to
"""
U_systematic.add_error_source(measurements=['Psi'],distribution='norm',params=[0,0.0295],source='b_Psi',units='deg')
U_systematic.add_error_source(measurements=['Phi'],distribution='norm',params=[0,0.025],source='b_Phi',units='deg')
U_systematic.add_error_source(measurements=['Pstat'],distribution='norm',params=[0,0.005],source='b_P_stat',units='psf')
U_systematic.add_error_source(measurements=['Ptot'],distribution='norm',params=[0,0.005],source='b_P_tot',units='psf')
U_systematic.add_error_source(measurements=['Baro'],distribution='norm',params=[0,0.025],source='b_P_baro',units='psf')
U_systematic.add_error_source(measurements=['Temp'],distribution='norm',params=[0,0.05],source='b_T',units='deg F')
U_systematic.add_error_source(measurements=['TempT'],distribution='norm',params=[0,0.05],source='b_T0',units='deg F')
U_systematic.add_error_source(measurements=['Theta'],distribution='norm',params=[0,0.009/2],source='b_Theta',units='deg')
U_systematic.add_error_source(measurements=['Qset','Qact'],distribution='norm',params=[0,0.14/2],source='b_Qcal',units='psf')
U_systematic.add_error_source(measurements=['NF'],distribution='norm',params=[0,0.3/2],source='b_NF',units='lbf')
U_systematic.add_error_source(measurements=['SF'],distribution='norm',params=[0,0.3/2],source='b_SF',units='lbf')
U_systematic.add_error_source(measurements=['AF'],distribution='norm',params=[0,0.06/2],source='b_AF',units='lbf')
U_systematic.add_error_source(measurements=['PM'],distribution='norm',params=[0,1.6/(2*12)],source='b_PM',units='ft.lbf')
U_systematic.add_error_source(measurements=['RM'],distribution='norm',params=[0,1.5/(2*12)],source='b_RM',units='ft.lbf')
U_systematic.add_error_source(measurements=['YM'],distribution='norm',params=[0,2.6/(2*12)],source='b_YM',units='ft.lbf')

# 1.4 Define random uncertainty for Variables of Interest (VOIs) using direct comparison of replicate data --OR-- define random uncertainty for elemental error sources 
U_random = utilities.U_random() # instantiate from random uncertainty class
replicate_data = {} # define the replicate data variable

if s_flag == 'P':
    # add random elemental error sources to propagate with MCM
    U_random.add_error_source(measurements=['Theta'],distribution='norm',params=[0,0.0088],source='s_Q-flex',units='deg')
    U_random.add_error_source(measurements=['Qset','Qact'],distribution='norm',params=[0,0.06],source='s_Q',units='psf')
    U_random.add_error_source(measurements=['NF'],distribution='norm',params=[0,0.077],source='s_NF',units='lbf')  
    U_random.add_error_source(measurements=['SF'],distribution='norm',params=[0,0.031],source='s_SF',units='lbf')
    U_random.add_error_source(measurements=['AF'],distribution='norm',params=[0,0.038],source='s_AF',units='lbf')
    U_random.add_error_source(measurements=['PM'],distribution='norm',params=[0,0.026],source='s_PM',units='in.lbf')
    U_random.add_error_source(measurements=['RM'],distribution='norm',params=[0,0.046],source='s_RM',units='in.lbf')
    U_random.add_error_source(measurements=['YM'],distribution='norm',params=[0,0.010],source='s_YM',units='in.lbf')
elif s_flag == 'DCR':
    # if direct comparison of replicate data is used, include that data here and select "DCR" for s_flag variable.

    # Example load of unexpanded (1 standard deviation) random uncertainty for VOIs in list of tuple format.  1st element of tuple is 
    # AOA [deg], 2nd is unexpanded random uncertainty for the VOI at that AOA.
    ### REPLACE WITH YOUR REPLICATE DATA RANDOM UNCERTANTIES
    replicate_data['CD'] = [(-5.1696176180857, 0.0001591001714773477), (-3.9038281366313283, 0.0002328149047544271), (-2.7015872529049942, 0.000255926596458534), (-1.4653745819689186, 0.00023525321670086246), (0.005571468152566158, 0.0002702930907591112), (1.4625085741663215, 0.0003232038437630706), (2.5085521866487785, 0.00022311370322637323), (3.782178415561224, 0.0001975598105961035), (5.26902419943794, 0.0002841372898124322), (6.3518042509958565, 0.0002677303360101057), (7.686353639591748, 0.0006978682620026499), (8.824421183061578, 0.0018574082451621157), (9.987800930228191, 0.0005839499314171011)]
    replicate_data['CL'] = [(-5.1696176180857, 0.0018709575732063172), (-3.9038281366313283, 0.001989294285249109), (0.005571468152566158, 0.0018901707900155947), (2.5085521866487785, 0.002383971631659089), (5.076555544067032, 0.0021087021830378167), (8.863682914191438, 0.0030536688041053295), (9.987800930228191, 0.0017020507250227065)]
    replicate_data['CY'] = [(-5.1696176180857, 0.00032560924956700045), (-3.9038281366313283, 0.0004352228201953415), (-1.3568822333596444, 0.0003968692037401731), (1.209110726991215, 0.00035873192471553704), (3.782178415561224, 0.0004089018778244675), (8.824421183061578, 0.0009034718080660066), (9.987800930228191, 0.0006620515508705179)]
    replicate_data['Cl'] = [(-5.1696176180857, 0.00033466213806678533), (-3.9038281366313283, 0.00042319849919467524), (-1.3568822333596444, 0.0004973849656935129), (2.5085521866487785, 0.00044684677618301113), (5.076555544067032, 0.0004626318393781211), (8.863682914191438, 0.0007213375783436461), (9.987800930228191, 0.00035401057494683356)]
    replicate_data['Cm'] = [(-5.1696176180857, 0.00030704703048325), (-3.9038281366313283, 0.00036977044074215496), (-2.648021800433413, 0.00035247560552901467), (-1.4653745819689186, 0.0003084334056462605), (0.005571468152566158, 0.00026100009765632454), (1.4625085741663215, 0.00025148703522785617), (2.5085521866487785, 0.0002796319422101068), (4.040723116790748, 0.00020143302159484578), (5.076555544067032, 0.00021223083654660666), (6.3518042509958565, 0.00020275119441069667), (7.686353639591748, 0.0005901874621180614), (8.863682914191438, 0.0007742594014534947), (9.987800930228191, 0.0004410929401040245)]
    replicate_data['Cn'] = [(-5.1696176180857, 4.122340605366101e-05), (-3.9038281366313283, 4.544863882020517e-05), (1.209110726991215, 4.573347711816445e-05), (4.040723116790748, 4.831319756132865e-05), (6.3518042509958565, 5.736730214603118e-05), (8.863682914191438, 0.0004810829960615475), (9.987800930228191, 0.0001758564407776527)]
    replicate_data['L_D'] = [(-5.1696176180857, 0.0), (1.4625085741663215, 0.1773918057176077), (5.076555544067032, 0.12134435377380748), (8.824421183061578, 0.25946335800892917), (9.987800930228191, 0.04019840127400916)]
    replicate_data['L12_D'] = [(-5.1696176180857, 0.0), (1.4625085741663215, 0.3096365149301357), (5.076555544067032, 0.1350892861725843), (8.824421183061578, 0.2608672426599938), (9.987800930228191, 0.03909758594744549)]
    replicate_data['L32_D'] = [(-5.1696176180857, 0.0), (2.5085521866487785, 0.13079875426434345), (5.076555544067032, 0.10869726389515232), (8.824421183061578, 0.25709954991187856), (9.987800930228191, 0.041904449763136516)]

#endregion


#region 2: Run MCM Simulation for propagation of measurand uncertainty
# !!! This region should not require any modification by the user.
RunData = utilities.MCM_sim(data, testinfo, U_systematic, U_random, s_flag, M)

if s_flag == 'DCR':
    # Add random uncertainty from direct comparison of replicate values if replicate data is available
    RunData = utilities.s_replicates(RunData,replicate_data)

# print time stamp
toc = time.time()
print('Total time elapsed = ' + str(toc-tic))
print('Run ' + str(testinfo['RunNum']) + ' complete')

# Calculate UPCs if desired 
if UPCs:
    utilities.UPCs(RunData, data, testinfo, U_systematic, U_random, s_flag, UPC_M, replicate_data)
    
# Save results
with open(outputfile, 'wb') as f:  
    pickle.dump([RunData,U_systematic,U_random],f)
print("Simulation data saved to file: {}".format(outputfile))

#endregion


#region 3: Plot results or run the WINDMONTE_GUI code

# Confidence level for expanding the uncertainty intervals is 95% by default and set when defining the VOI class in utilities.py

# Running GUI is the default
os.system("python WINDMONTE_GUI.py")

# or plot directly using plot behaviors for the TestRun, DataPoint, or VOI variable class
""" Plot_VOIs = ['CD', 'CY','CL','Cl','Cm','Cn']
RunData.plot_errorbars('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',Plot_VOIs,ncols=3)
RunData.plot_U_VOI('AlphaC',['L_D','L12_D','L32_D'],ncols=3) 
plt.show()"""
#endregion

#from . import config
import pydoc
import numpy as np
import sys

'''
The tconst.py module contains variables for test-specific constants. 
These include (but are not limited to) model geometries, 
boundary-correction constants, boundary-layer profile parameters and 
installation information.

NOTE: All parameters defined below must be of type 'float'. This 
means that a decimal point must exist on the right-hand-side of the 
equals sign

INPUTS:
	
	N/A

OUTPUTS:
	
	Variables contained in this module are accessed as 
	tconst.VARIABLE_NAME in any code which imports tconst.py
	
'''

#
# Distance from Moment Reference Center (MRC) to Pivot Point Location (PPL)
# 
XMRC = 4.4122 					#[in]		Longitudinal Moment Reference Center, (+Forward, BMC Location)
YMRC = 0.0000					#[in]		Lateral Moment Reference Center,      (+Starboard, BMC Location)
ZMRC = -1.604					#[in]		Vertical Moment Reference Center,     (+Down, BMC Location)

XMRC = -5					#[in]		Longitudinal Moment Reference Center, (+Forward, BMC Location)
YMRC = 0.0000					#[in]		Lateral Moment Reference Center,      (+Starboard, BMC Location)
ZMRC = 0					#[in]		Vertical Moment Reference Center,     (+Down, BMC Location)

X_CG = 0.188
Y_CG = 0
Z_CG = -0.08
#
# Distance from Pivot Point Location (PPL) to Balance Moment Center (BMC)
# 
XPPL = +0.000						#[in]		Longitudinal displacement from PPL to BMC 	(+Forward)
YPPL = +0.000						#[in]		Lateral displacement from PPL to BMC		(+Starboard)
ZPPL = +0.000						#[in]		Vertical displacement from PPL to BMC		(+Down)

#
# Rotation Settings
#
PHI_OFFSET = 0.						#[deg]		Angular roll offset between mount and model body with motion system(s) set to 0 (adds to PHI. Positive implies model is rolled positive relative to mount)
THETA_OFFSET = -10.					#[deg]		Angular pitch offset between mount and model body with motion system(s) set to 0 (adds to THETA)
PSI_OFFSET = 0.						#[deg]		Angular yaw offset between mount and model body with motion system(s) set to 0 (adds to PSI)
ROLL_KEY = 'Phi'					#[str]		What ROLL ANGLE key should I use to rotate between inertial frame and mount frame? [also condition-matching?]
PITCH_KEY = 'Theta'					#[str]		What PITCH ANGLE key should I use to rotate between inertial frame and mount frame? [also condition-matching?]
YAW_KEY = 'Psi'						#[str]		What YAW ANGLE key should I use to rotate between inertial frame and mount frame? [also condition-matching?]


#
# Other dimensions
#
SPLITTER_HEIGHT=4.0000				#[in]		Splitter Plate height 
SCALE_FACTOR = 300.					#[N/A]		Scale factor of model
ZREF = 10.0 						#[m]		Full scale customer-specified reference height
BL_PROFILE ="NPD"					#[N/A]		Specified boundary layer profile shape
BL_VARY = 0.04						#[N/A]		Percent variation allowed in BL profile setup
n = 0.125							#[N/A]		Exponent needed for ABS/API BL profiles. This can remain even if an NPD profile

# 
# PSI/sting/other? Configuration
# 
PSI_SCAN = ['SN32134','SN32135']	#[N/A]		List of pressure scanners in order from Sys 8400. This should be the SN key for the scanner
STING = 'SN40x150'#'SN4495x150'#	'SN2975x125'#			#[N/A]		'Serial' number for sting used during test (not used if not a sting-mounted model or internal balance test): !!!!Confirm 40 inch sting for test 1930 with blade!!!!
sting_offset = 0
QCUTOFF = 0.05						#[psf]	 	Dynamic pressure cutoff (Assume wind-off if less than QCUTOFF)
QMIN    = 0.1						#[psf]	 	Wind-off Qact setting if Qact falls below QCUTOFF

#
# Correction Constants
#
CREFF = 0.0					#[ft]		Reference flaperon chord dimension
CREFR = 0.0					#[ft]		Reference flaperon chord dimension
SREFF = 0.0				    #[ft**2]	Reference flaperon area dimension
SREFR = 0.0				    #[ft**2]	Reference flaperon area dimension


CTUNN = 68.0						#[ft**2]	Tunnel Cross-Sectional Area
CREF = 1.045   #4.75/6.0						#[ft]		Reference chord dimension
BREF = 7.35   #6.0						#[ft]		Reference span dimension
SREF = 7.2   #4.75						#[ft**2]	Reference planform area dimension
VB = 0.3391	    					#[ft**3]	Model body volume
K3B = 0.905							#[N/A]		Solid-blockage correction parameter (fuselage)
TAU1B = 0.87						#[N/A]		Solid-blockage correction parameter (fuselage)
VW = 0.5752		    				#[ft^3]		Model wing volume
K1W = 1.0							#[N/A]		Solid-blockage correction parameter (wing)
TAU1W = 0.88					#[N/A]		Solid-blockage correction parameter (wing)
VN = 0.0	    					#[ft**3]	Model nacelle volume
K3N = 0.0						#[N/A]		Solid-blockage correction parameter (nacelle)
TAU1N = 0.0						#[N/A]		Solid-blockage correction parameter (nacelle)
VT = 0.0399	    					#[ft**3]	Model tail volume
K1T = 1.010						#[N/A]		Solid-blockage correction parameter (tail)
TAU1T = 0.855						#[N/A]		Solid-blockage correction parameter (tail)
DELTA = 0.126						#[N/A]		Streamline curvature/normal downwash correction parameter
TAU2W = 0.075						#[N/A]		Streamline curvature/normal downwash correction parameter
TAU2T = 0.052						#[N/A]		Streamline curvature/normal downwash correction parameter
DCMDD = -0.07						#[1/deg]	Tail effectiveness
DA_UP = 1.00						#[deg]		upflow bias in test section due to model presence


"""
Uncertainty PDF Info 
Required items in dict: variable, type (B for systematic and S for random), source name, distribution type ('norm','uniform','triang','percentnorm'), and items for specific PDF type below

Normal Distribution:
'PDF': 'norm' 
'STD': X, where X is 1 standard deviation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)

Uniform Distribution:
'PDF': 'uniform'
'a': left half of the distribution, mean-left limit (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform)
'b': right half of the distribution, right limit-mean

Triangular Distribution
'PDF': 'triang'
'scale': X, where X is the value of half the uncertainty interval (one side)
'c': Y, where Y is the shape parameter 0<=c<=1 that determines where the apex of the distribution occurs
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html#scipy.stats.triang

Lognormal Distribution
'PDF': 'lognorm'
'STD': X
'x': Y
's': Z
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm

Percent Full-Scale Normal Distribution
'PDF': 'norm' 
'STD': X, where X is 1 standard deviation at full scale
'FS': Y, where Y is the nominal value at full scale
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
"""



k = 2  # Uncertainty expansion factor (1 = 1 STD, 68% confidence; 2 = 2 STD, 95% confidence, 3 = 3 STD, 99.7% confidence)
PDFformat = 'PDF'  # enter "PDF" to save in PDF format in variables, "iCDF" to convert all PDFs to inverse CDFs.
q_iCDF = np.linspace(0.001,0.999,250)  # array for iCDF inputs to interpolate

#Null_offset = {'NF':1.458,'AF':0.1726,'PM':-0.56867,'RM':0.02492,'YM':0.034063,'SF':0.106118}
Null_offset = {'NF':1.458,'AF':-1.7,'PM':-0.56867,'RM':0.02492,'YM':0.034063,'SF':0.106118}

conv_criteria = {}
# List out any convergence criteria to be used
conv_criteria['CD'] = {}
conv_criteria['CD']['mean'] = 0.00005
conv_criteria['CD']['U'] = 0.00005
conv_criteria['CD']['low'] = 0.00005
conv_criteria['CD']['high'] = 0.00005
conv_criteria['CY'] = {}
conv_criteria['CY']['mean'] = 0.0005
conv_criteria['CY']['U'] = 0.0005
conv_criteria['CY']['low'] = 0.0005
conv_criteria['CY']['high'] = 0.0005
conv_criteria['CL'] = {}
conv_criteria['CL']['mean'] = 0.0005
conv_criteria['CL']['U'] = 0.0005
conv_criteria['CL']['low'] = 0.0005
conv_criteria['CL']['high'] = 0.0005
conv_criteria['Cl'] = {}
conv_criteria['Cl']['mean'] = 0.0005
conv_criteria['Cl']['U'] = 0.0005
conv_criteria['Cl']['low'] = 0.0005
conv_criteria['Cl']['high'] = 0.0005
conv_criteria['Cm'] = {}
conv_criteria['Cm']['mean'] = 0.0005
conv_criteria['Cm']['U'] = 0.0005
conv_criteria['Cm']['low'] = 0.0005
conv_criteria['Cm']['high'] = 0.0005
conv_criteria['Cn'] = {}
conv_criteria['Cn']['mean'] = 0.0005
conv_criteria['Cn']['U'] = 0.0005
conv_criteria['Cn']['low'] = 0.0005
conv_criteria['Cn']['high'] = 0.0005

obj_metrics = [
    ['Point','Point','-', 9,'int'],
    ['CDmin', 'CDmin', '[-]',16,5],
	['CLmax', 'CLmax', '[-]',16,5],
    ['CLmindrag', 'CLmindrag', '[-]',16,5],
    ['LoD_max', 'L/Dmax', '[-]',14,5],
    ['L32oD_max', 'L^(3/2)/D max', '[-]',16,5],
    ['L12oD_max', 'L^(1/2)/D max', '[-]',16,5],
    ['Cm0', 'Cm0', '[-]',12,5],
    ['dCm_dA', 'dCm/dAlpha', '[-]',14,5],
    ['dCL_dAlpha', 'dCL/dAlpha', '[-]',16,5]]

if __name__ == "__main__":
	if config.help: pydoc.help('tconst'); raise SystemExit

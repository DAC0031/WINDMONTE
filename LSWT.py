
""" 
This file is a simplification of the evaluation function used at the Oran Nicks LSWT.  Portions of the code, such as static tares and data input, have been simplified greatly for the 
puposes of demonstrating WINDMONTE only and are no longer accurate.  This elimates the need to attach numerous static tare and redux files.  This entire file should be deleted and replaced
with the appropriate DREs for the facility using WINDMONTE.
"""

import re
import tconst
import numpy as np
import copy

def DREs(data,G):

    D = copy.deepcopy(data)
    # evaluation function that maps measured inputs and constants to DRE outputs (wind-frame aerodynamic coefficients and performance VOIs)

    # check to make sure simulated error doesn't send invalid data
    for point in range(len(D)):
        for key in D[point].keys():
            if key in ['Qact','Qset','Ptot','Baro','Mach','rho']: D[point][key] = np.abs(D[point][key])

    for i in range(len(D)):
        D[i]['ThetaB'] = D[i]['Theta']+tconst.THETA_OFFSET
        D[i]['Alpha'] = D[i]['ThetaB']

    # Define body-fixed aero angles: Gamma=0 (because roll is aerodynamically insignificant), Alpha, Beta with offsets incorporated
    COMPUTE_AERO_ANGLES(D,G)

    # Determine solid blockage parameters based on config code
    CONFIG_LOGIC(G)

    if 'AF' in list(D[0].keys()):
        PERFORM_IB_FIT = 1

        # !!!! IMPORTANT !!!
        # This line bypassess the original static tare code and makes over-simplifications for the example code posted on GitHub.  
        # This is purely to eliminate the need of attaching all the static tare code and data to this MCM implementation, this eval function should not be used for obtaining actual results
        if True:  
            PERFORM_IB_FIT = 0
            W = np.sqrt(D[0]['NF']**2+D[0]['AF']**2)
            for i in range(len(D)):
            # Subtract gravity loads in BALANCE frame
                offset = 0
                D[i]['AF'] -= W*np.sin((np.pi/180)*(D[i]['Theta']))
                D[i]['NF'] += W*np.cos((np.pi/180)*(D[i]['Theta']))
                D[i]['PM'] += (W*(tconst.X_CG*np.cos(D[i]['ThetaB']*np.pi/180))/12)
        
        # Subtract gravity contribution (this is performed in the BALANCE frame)
        if PERFORM_IB_FIT:
            D,G = static.FIT_IB(D, G)

        
        # Calculate coefficients 
        CALCULATE_COEFFICIENTS(D,frame='body')
        
        # Rotate BALANCE-fixed data to BODY frame and WIND frame -- wind-frame rotation only needed so data can be printed to stat files
        for i in range(len(D)):
            CFXBAL,CFYBAL,CFZBAL = -D[i]['CAB'],+D[i]['CYB'],-D[i]['CNB']
            CMXBAL,CMYBAL,CMZBAL = +D[i]['ClB'],+D[i]['CmB'],+D[i]['CnB']
            
            #SUBROUTINE
            R_BAL_TO_BODY = np.dot(MOUNT_TO_BODY_MATRIX(D[i]),BALANCE_TO_MOUNT_MATRIX(D[i]))
            CFXBAL = np.float64(CFXBAL)
            CFYBAL = np.float64(CFYBAL)
            CFZBAL = np.float64(CFZBAL)
            CXB,CYB,CZB = np.dot(R_BAL_TO_BODY,[CFXBAL,CFYBAL,CFZBAL])
            D[i]['CAB'],D[i]['CYB'],D[i]['CNB'] = -CXB,CYB,-CZB
            D[i]['ClB'],D[i]['CmB'],D[i]['CnB'] = np.dot(R_BAL_TO_BODY,[CMXBAL,CMYBAL,CMZBAL])
            
            #SUBROUTINE
            R_BODY_TO_WIND = np.dot(
                np.transpose(MAKE_E_MATRIX([('z',-D[i]['Beta']),('y',0),('x',0)])),
                np.transpose(MAKE_E_MATRIX([('z',0),('y',D[i]['Alpha']),('x',0)]))
            )
            CXW, CYW, CZW = np.dot(R_BODY_TO_WIND,[-D[i]['CAB'],+D[i]['CYB'],-D[i]['CNB']])
            D[i]['CD'],D[i]['CY'],D[i]['CL'] = -CXW, CYW, -CZW
            D[i]['Cl'],D[i]['Cm'],D[i]['Cn'] = np.dot(R_BODY_TO_WIND,[+D[i]['ClB'],+D[i]['CmB'],+D[i]['CnB']])
        

        # Transfer moments using body-fixed load data (output is transferred moments in body frame only, no change to wind- or body-fixed FORCES)
        TRANSFER_MOMENTS(D)
        
        # Only body-fixed loads are correct, need to re-rotate for updates to wind-fixed loads
        for i in range(len(D)): # Rotate body-fixed force data to wind frame (moments already computed)
            #SUBROUTINE
            R_BODY_TO_WIND = np.dot(
                np.transpose(MAKE_E_MATRIX([('z',-D[i]['Beta']),('y',0),('x',0)])),
                np.transpose(MAKE_E_MATRIX([('z',0),('y',D[i]['Alpha']),('x',0)]))
            )
            CXW, CYW, CZW = np.dot(R_BODY_TO_WIND,[-D[i]['CAB'],+D[i]['CYB'],-D[i]['CNB']])
            D[i]['CD'],D[i]['CY'],D[i]['CL'] = -CXW, CYW, -CZW
            D[i]['Cl'],D[i]['Cm'],D[i]['Cn'] = np.dot(R_BODY_TO_WIND,[+D[i]['ClB'],+D[i]['CmB'],+D[i]['CnB']])


    # Perform wall corrections on blockage-corrected data (whether T&I test or not)
    # WALL_CORREX applies to WIND-FRAME data, BLOCKAGE is agnostic
    D = wallcorrex(G,blockage(G['eSB'],D))

    # We performed corrections in the wind frame, rotate back to body frame using Alpha, -Beta
    for i in range(len(D)):
        R_WIND_TO_BODY = np.dot(
            MAKE_E_MATRIX([('z',0),('y',D[i]['AlphaC']),('x',0)]),
            MAKE_E_MATRIX([('z',-D[i]['Beta']),('y',0),('x',0)])
        )
        CXB, CYB, CZB = np.dot(R_WIND_TO_BODY,[-D[i]['CD'],+D[i]['CY'],-D[i]['CL']])
        D[i]['CAB'],D[i]['CYB'],D[i]['CNB'] = -CXB, +CYB, -CZB
        D[i]['ClB'],D[i]['CmB'],D[i]['CnB'] = np.dot(R_WIND_TO_BODY,[+D[i]['Cl'],+D[i]['Cm'],+D[i]['Cn']])

    # Generate "true" coefficients prior to printing
    for i in range(len(D)):
        for coef_key in ['CD','CY','CL','CAB','CYB','CNB']:
            D[i][coef_key] /= tconst.SREF
        for coef_key in ['Cm','CmB']:
            D[i][coef_key] /= (tconst.SREF*tconst.CREF)
        for coef_key in ['Cl','Cn','ClB','CnB']:
            D[i][coef_key] /= (tconst.SREF*tconst.BREF)

    for i in range(len(D)):
        if D[i]['CL'] < 0:
            D[i]['L_D'] = 0
            D[i]['L12_D'] = 0
            D[i]['L32_D'] = 0
        else:
            D[i]['L_D'] = D[i]['CL']/D[i]['CD']
            D[i]['L12_D'] = D[i]['CL']**(1/2)/D[i]['CD']
            D[i]['L32_D'] = D[i]['CL']**(3/2)/D[i]['CD']

    return D


##########################################################
# Defining additional functions used inside DREs.eval()

def CALCULATE_COEFFICIENTS(d,frame='wind'):
    '''
    Generate pseudo-coefficients in square-length and cubic-length units;
    simply divide all forces and moments by q
    '''
    
    if frame == 'wind':
        LOT1 = [('Fx','CD'),('Fy','CY'),('Fz','CL'),('Mx','Cl'),('My','Cm'),('Mz','Cn')]
        LOT2 = [('Fx_stdev','CD_stdev'),('Fy_stdev','CY_stdev'),('Fz_stdev','CL_stdev'),('Mx_stdev','Cl_stdev'),('My_stdev','Cm_stdev'),('Mz_stdev','Cn_stdev')]
        LLOT = [LOT1,LOT2]
    elif frame == 'body':
        LLOT = [('AF','CAB'),('SF','CYB'),('NF','CNB'),('RM','ClB'),('PM','CmB'),('YM','CnB')]
    else: 
        raise Exception("routines.CALCULATE_COEFFICIENTS: Error on frame spec")
        # iterate over list of tuples to associate coeffs with dimensional loads
    
    for el in d:
        for tupleSet in LLOT:
            for dim,nondim in LLOT:
                if el['Qact'] == 0: 
                    el[nondim] = el[dim]/(el['Qact']+0.00001)
                else: 
                    el[nondim] = el[dim]/el['Qact']
                # Drag, lift are negative Fx, Fz	
                if nondim in ['CD','CD_stdev','CL','CL_stdev']:
                    el[nondim] *= -1
        
    return

def CONFIG_LOGIC(g):
    '''
    change that CONFIG_LOGIC describes how to define boundary-correction
    parameters based on configuration codes specified in 
    Configuration File - V1.xlsx
    
    DEPENDENCIES: re, Common.tconst
    '''
    # Compute sb correction for model body; specific to current test
    g['eSBB'] = tconst.K3B*tconst.TAU1B*tconst.VB/tconst.CTUNN**(1.5)

    # Initialize auxiliary solid blockage parameters
    g['eSBW'] = 0
    g['eSBN'] = 0
    g['eSBT'] = 0
    
    # Initialize wall-correction parameters
    g['delta'] = 0;g['tau2w'] = 0;g['tau2t'] = 0
    g['dCMdd'] = 0
        
    # Compute/assign correction parameters based on model config
    # check for wing
    if re.search(r'W',g['config']): 
        g['eSBW'] = tconst.K1W*tconst.TAU1W*tconst.VW/tconst.CTUNN**(1.5)
        g['delta'],g['tau2w']=tconst.DELTA,tconst.TAU2W

    # check for tail
    if re.search(r'V',g['config']):
        g['tau2t']=tconst.TAU2T
        g['eSBT'] = tconst.K1T*tconst.TAU1T*tconst.VT/tconst.CTUNN**(1.5)
        g['dCMdd'] = tconst.DCMDD
        
    # check for nacelles
    if re.search(r'P',g['config']):
        g['eSBN'] = tconst.K3N*tconst.TAU1N*tconst.VN/tconst.CTUNN**(1.5)
        # raise exception if nacelles installed without a wing
        if not re.search(r'W',g['config']):
            raise Exception("ERROR: NACELLES INSTALLED WITHOUT A WING?!")

    g['eSB'] = g['eSBB']+g['eSBW']+g['eSBT']+g['eSBN']

    return 

def wallcorrex(G_DICT,WAMC_DATA):
	''' 
	Perform streamline curvature, normal downwash, and tail
	upwash boundary corrections to wind-axis, model-centered data
	
	INPUTS: 
		G_DICT		: Dictionary containing test constants
		WAMC_DATA	: List of dictionaries containing model-centered, 
					  wind-frame coefficient data
		
	OUTPUTS :
		Ci			: Corrected force and moment coefficients added to DATA list
		
	'''
	g,d = G_DICT,WAMC_DATA
	da = 0.;CLw = 0.

	for i in range(len(d)):
		# Logic to find CL,wing and dCL/d(alpha)
		#
		#
		#
		
		# Use thin-airfoil theory until more sophisticated logic
		AR = tconst.BREF**2/tconst.SREF
		CLw = 2*np.pi*np.deg2rad( d[i]['Alpha'])*AR/(AR+2.)
		dCLda = 2*np.pi*AR/(AR+2.) # [1/deg]
		
		#if WingData[i][Qc] < qCutoff: dCLda = 0 # dCLda ~ INF between wind-off and wind-on zeros
		#print "Target Alpha ", CLw, dCLda

		d[i]['dASC'] = np.rad2deg(g['delta']*g['tau2w']*tconst.SREF/tconst.CTUNN*CLw)		#[deg] dAlpha due to streamline curvature
		d[i]['dATU'] = np.rad2deg(g['delta']*g['tau2t']*tconst.SREF/tconst.CTUNN*CLw)		#[deg] dAlpha due to tail upwash
		d[i]['dAND'] = np.rad2deg(g['delta']*tconst.SREF/tconst.CTUNN*CLw)					#[deg] dAlpha due to normal downwash
		dAlphaW = d[i]['dASC']+d[i]['dAND']											#[deg] dAlpha due to walls (SC and ND)

		#Pitch Alignment Correction (4ci)
		da = tconst.DA_UP + dAlphaW

		#Data Alignment Correction (4cii)
		#d[i]['Alpha'] += da 
		d[i]['AlphaC'] = d[i]['Alpha'] + da 
		
		# Define temp values for comparion after alignment correction
		tempCD,tempCL = d[i]['CD'],d[i]['CL']
		
		R2 = np.transpose(MAKE_E_MATRIX([('y',da)]))
		CXW,CYW,CZW = np.dot(R2,[-d[i]['CD'],d[i]['CY'],-d[i]['CL']])
		d[i]['CD'],d[i]['CY'],d[i]['CL'] = -CXW,CYW,-CZW
		
		# Add dC to data structure (dC from alignment correction)
		d[i]['dCDalgn'] = d[i]['CD']-tempCD
		d[i]['dCLalgn'] = d[i]['CL']-tempCL

		## Wall Corrections (4ciii)
		#Streamline curvature
		d[i]['dCLsc'] = -d[i]['dASC']*dCLda
		d[i]['CL']   += d[i]['dCLsc']
		d[i]['dCMsc'] = -0.25*d[i]['dCLsc']
		d[i]['Cm']   += d[i]['dCMsc']

		#Tail effectiveness (tail upwash)
		d[i]['dCMtu'] = -g['dCMdd']*d[i]['dATU']
		d[i]['Cm']   += d[i]['dCMtu']

		#Normal Downwash (come back in and use sine/cosine??)
		d[i]['dCDnd'] = CLw*np.deg2rad(d[i]['dAND'])
		d[i]['CD']   += d[i]['dCDnd']		
		
	return d

def blockage(eSB,DATA):
	'''
	Correct the test section dynamic pressure (Qact) based on 
	model/test section geometry and drag. Operations are performed
	on static-tared, T&I-removed, wind-frame pseudo-coefficient data (EPAs). 
	'Uncorrected' pseudo-coefficients and Qcor are added to the DATA
	dictionary. 
	
	INPUTS: 
		eSB		: Solid blockage epsilon (dV/V)
		DATA	: List of dictionaries containing run data
		
	OUTPUTS :
		eWB		: wake blockage epsilon added to DATA list
		Qcor	: corrected dynamic pressure added to DATA list
		Ci		: uncorrected force and moment coefficients added to DATA list
		
	'''
	
	# Operate on DATA dict
	for i in range(len(DATA)):
				
		DATA[i]['eWB'] = DATA[i]['CD']/4./tconst.CTUNN # Compute wake blockage correction
		
		# Compute corrected dynamic pressure
		DATA[i]['Qcor'] = (DATA[i]['Qact']*(1.+eSB+DATA[i]['eWB'])**2)
		
		# Compute uncorrected pseudo-coefficients (non-dim by Qcor)
		for coef in ['CD','CY','CL','Cl','Cm','Cn']:
			DATA[i][coef] *= DATA[i]['Qact']/DATA[i]['Qcor']
	
	return DATA

def TRANSFER_MOMENTS(DATA,frame="body"):
    '''
    TRANSFER_MOMENTS transfers body-fixed moments from balance 
    moment center (BMC) to model reference center (MRC) using 
    geometry data from tconst.py.
    
    After body-fixed moments are transferred to MRC, they are NOT
    rotated back to wind frame before being returned to calling 
    function.
    
    INPUTS:
        
        DATA 
                List of dictionaries containing body-fixed forces and 
                moments.
                
                DATA must contain the following fields:
                    
                    CAB,CYB,CNB
                        Body-fixed axial, side, and normal 
                        pseudo-coefficients (FORCE/Qact)
                    
                    ClB,CmB,CnB
                        Body-fixed roll, pitch, and yaw 
                        pseudo-coefficients (MOMENT/Qact)

        *frame		The default for this optional argument is "body"
                If the user instead passes "wind" for this arument,
                xPPL, yPPL, zPPL will be ignored, and xMRC, yMRC, zMRC
                will be taken as the wind-fixed position vector 
                ORIGINATING at BMC and TERMINATING at MRC
    
    OUTPUTS:
    
        DATA 
                List of dictionaries containing moment-transferred 
                body-fixed forces and moments.
    
    '''
    xMRC,yMRC,zMRC = tconst.XMRC/12.,tconst.YMRC/12.,tconst.ZMRC/12.
    xPPL,yPPL,zPPL = tconst.XPPL/12.,tconst.YPPL/12.,tconst.ZPPL/12.

    if frame == 'body':
        # Perform moment transfers using body-fixed pseudo-coefficient (EPA) data
        for i in range(len(DATA)):
            
            #x0 = xMRC + xPPL*np.cos(np.deg2rad(DATA[i]['Theta']))
            #y0 = yMRC + yPPL
            #z0 = zMRC + xPPL*np.sin(np.deg2rad(DATA[i]['Theta']))
            
            ## Generic: 
            x0 = xMRC + xPPL*np.cos(np.deg2rad(DATA[i]['Alpha'])) - zPPL*np.sin(np.deg2rad(DATA[i]['Alpha']))
            y0 = yMRC + yPPL
            z0 = zMRC + xPPL*np.sin(np.deg2rad(DATA[i]['Alpha'])) + zPPL*np.cos(np.deg2rad(DATA[i]['Alpha']))
            
            # HOPEFULLY THIS IS OBSOLETE
            #if DATA[0]['TargetPhi']==180.:
            #	y0 *= -1.
            #	z0 *= -1.
            
            DATA[i]['ClB'] += (-y0*DATA[i]['CNB']-z0*DATA[i]['CYB'])
            DATA[i]['CmB'] += (-z0*DATA[i]['CAB']+x0*DATA[i]['CNB'])
            DATA[i]['CnB'] += (+x0*DATA[i]['CYB']+y0*DATA[i]['CAB'])
    
    elif frame == 'wind':
        # Perform moment transfers using wind-fixed pseudo-coefficient (EPA) data
        for i in range(len(DATA)):
            x0 = xMRC
            y0 = yMRC
            z0 = zMRC
            
            
            DATA[i]['Cl'] += (-y0*DATA[i]['CL']-z0*DATA[i]['CY'])
            DATA[i]['Cm'] += (-z0*DATA[i]['CD']+x0*DATA[i]['CL'])
            DATA[i]['Cn'] += (+x0*DATA[i]['CY']+y0*DATA[i]['CD'])

    else:
        raise Exception("Error: bad call to routines.TRANSFER_MOMENTS")

    #return DATA
    return

def COMPUTE_AERO_ANGLES(d,g):
    '''
    inputs.PARSE_CML outputs force-and-moment data in the 'balance' frame:
    for internal and external balance, 'balance frame' means 'frame of the balance' and nothing more.

    For example, WB-57's balance frame rotated 10-deg about the pitch axis relative to the wind-frame *when all motion systems are set to 0*
    A T&I model is mounted with a 180-deg roll offset relative to the wind-frame when all motion systems are set to 0 for the inverted portion of testing

    Once we're outside of inputs.PARSE_CML and after we've printed the raw...txt file with balance-frame data,
    align our data to the body, but *not the body-fixed coordinate system*. Some other intermediate coordinate system, where it's as though there was no mounting offset(s)
    
    # After running through this routine, the definitions of relevant angles:
    # Phi, Theta, Psi: Angles of MOUNT datum relative to INERTIAL FRAME with all motion systems set to 0
    # PhiB, ThetaB, PsiB: Angles of MODEL BODY relative to in INERTIAL FRAME with all motion systems set to 0, IF MODEL AND BALANCE WERE ALIGNED
    
    '''
    for thispoint in d:
    
        # [Phi, Theta, Psi] are angles of mount relative to inertial frame
        # [PhiB, ThetaB, PsiB] are angles of model body relative to inertial frame
        thispoint['PhiB'] = (thispoint['Phi']+tconst.PHI_OFFSET)%360
        thispoint['ThetaB'] = thispoint['Theta']+tconst.THETA_OFFSET
        thispoint['PsiB'] = thispoint['Psi']+tconst.PSI_OFFSET
        
        # rotate wind vector to body frame (starts in inertial frame which is same as xbal balance frame but NOT NECESSARILY same as ibal balance frame)
        # inertial -> balance, balance -> mount, mount -> body		
        GRAV2BOD = np.dot(
            MOUNT_TO_BODY_MATRIX(thispoint),
            np.dot(
                BALANCE_TO_MOUNT_MATRIX(thispoint),
                INERTIAL_TO_BALANCE_MATRIX(thispoint)
            )
        )
        
        windVectorBody = np.dot(GRAV2BOD,[-1,0,0])
        u,v,w = windVectorBody
        
        thispoint['Gamma'] = 0
        thispoint['Alpha'] = thispoint['ThetaB']
        thispoint['Beta'] = -1.*np.rad2deg(np.arctan(v/np.sqrt(u**2+w**2)))
        
        #thispoint['Gamma'] = 0
        #thispoint['Alpha'] = np.rad2deg(np.arctan2(w,u))
        #thispoint['Beta'] = -1.*np.rad2deg(np.arctan2(v,np.sqrt(u**2+w**2)))
    
    return

def MAKE_E_MATRIX(rotations):
	'''
	MAKE_E_MATRIX composes a rotation matrix designed to transer a 
	force vector to a new coordinate frame.
	
	INPUTS:
		
		rotations		
						a list of (axis,angle) tuples that specify a 
						rotation axis and magnitude which can be of 
						arbitrary length. The first tuple is the first 
						rotation, followed by the second tuple, etc. The 
						axis names can be any of x,y,z,X,Y,Z and the 
						angles are in degrees.
	OUTPUTS:
	
		e 				
						a compilation rotation matrix composed of the 
						multiplication of all the individual rotation 
						matrices (computed in ROTATE) used to transform 
						forces between reference frames.
	
	'''
	
	e = np.array([ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ])
	

	for axis,angle in rotations: 
		e = ROTATE(e,axis,angle)	
	return e
		
def ROTATE(init,axis,ang):
	'''
	INPUTS: 
		
		init	 			
							When called from MAKE_E_MATRIX, init is the 
							matrix e which begins as the identity matrix 
							and for each succesive loop	is modified to 
							be the compilation rotation matrix to that 
							point.
		
		axis 				
							axis specified by MAKE_E_MATRIX to be 
							x,y,z,X,Y,Z
		
		ang 				
							angle in degrees by which to rotate around 
							axis
	
	OUTPUTS:	
							A 3x3 matrix computed representing the 
							complete set of rotations specified in 
							rotations, the list of tuples passed to 
							MAKE_E_MATRIX
							
	'''
	
	if len(init) != 3 or len(init[0]) != 3:
		raise Exception("\nMatrix to be rotated isn't 3 rows (rotation.ROTATE).\n")
	
	c = np.cos(np.radians(float(ang)))
	s = np.sin(np.radians(float(ang)))
		
	r = np.array([ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ])
	#Make the individual rotation matrix
	if axis == 'x' or axis == 'X':
		r[1][1]=r[2][2]=c
		r[2][1]=-s
		r[1][2]=s
	elif axis == 'y' or axis == 'Y':
		r[0][0]=r[2][2]=c
		r[0][2]=-s
		r[2][0]=s
	elif axis == 'z' or axis == 'Z':
		r[0][0]=r[1][1]=c
		r[1][0]=-s
		r[0][1]=s
	else: raise Exception("\nBad rotation axis specification (rotation.ROTATE)\n")

	#If called as stand alone print the r matrix
	#It is multiplied by the identity matrix to make the output look readable
	if __name__=="__main__":
		I = [ [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0] ]
		print("\nr-matrix for %(ang)d deg rotation about %(axis)s axis:" %locals())
		print(np.dot(r,I),"\n")
	return np.dot(r,init)

def INERTIAL_TO_BALANCE_MATRIX(dict_element):
	'''
	COMMENTS
	'''
	if 'Fx' in list(dict_element.keys()) and 'AF' not in list(dict_element.keys()):
		R = np.eye(3)
		# if config.debug:print("rotation.INERTIAL_TO_BALANCE_MATRIX, XBAL, returning [I]")
	elif 'AF' in list(dict_element.keys()):
		R = MAKE_E_MATRIX([('z',dict_element[tconst.YAW_KEY]),('y',dict_element[tconst.PITCH_KEY]),('x',dict_element[tconst.ROLL_KEY])])
		# if config.debug:
			# print(
			# "rotation.INERTIAL_TO_BALANCE_MATRIX, IBAL, returning R321(%.0f,%.0f,%.0f)"
			# %(dict_element[tconst.YAW_KEY],dict_element[tconst.PITCH_KEY],dict_element[tconst.ROLL_KEY])
			# )
	else:
		print("rotation.py: Unclear how attitude angles are defined, returning Identity")
		R = np.eye(3)
	
	return np.array(R)

def BALANCE_TO_MOUNT_MATRIX(dict_element):
	'''
	COMMENTS
	'''
	if 'Fx' in list(dict_element.keys()) and 'AF' not in list(dict_element.keys()):
		# The lower turntable can only rotate in yaw with respect to the external balance
		R = MAKE_E_MATRIX([('z',dict_element[tconst.YAW_KEY]),('y',0),('x',0)])
		# if config.debug:print("rotation.BALANCE_TO_MOUNT_MATRIX, XBAL, returning R321(%.0f,0,0)"%dict_element[tconst.YAW_KEY])
	elif 'AF' in list(dict_element.keys()):
		# this is true as long as the internal balance is mounted in alignment with HARS (mount)
		# if not, angles may need to be defined as negative of what we'd initially think?
		R = np.eye(3)
		# if config.debug:print("rotation.BALANCE_TO_MOUNT_MATRIX, IBAL, returning [I]")
	else:
		print("rotation.py: Unclear how attitude angles are defined, returning Identity")
		R = np.eye(3)
		
	return np.array(R)
	
def MOUNT_TO_BODY_MATRIX(dict_element):
	'''
	COMMENTS
	'''
	if 'Fx' in list(dict_element.keys()) and 'AF' not in list(dict_element.keys()):
		# for xbal test, the mount (lower turntable) is responsible for yaw motion. 
		# An accessory (linear actuator) allows us to remote-control the THETA_OFFSET and it's this value(s) we report in the Pitch column of Config.xlsx
		# Any roll angle for an external balance model is by definition a PHI_OFFSET but is reported in the Roll column of Config.xlsx.
		R = MAKE_E_MATRIX([('z',tconst.PSI_OFFSET),('y',dict_element[tconst.PITCH_KEY]),('x',dict_element[tconst.ROLL_KEY])])
		# if config.debug:
			# print(
			# "rotation.MOUNT_TO_BODY_MATRIX, XBAL, returning R321(%.0f,%.0f,%.0f)"
			# %(tconst.PSI_OFFSET,dict_element[tconst.PITCH_KEY],dict_element[tconst.ROLL_KEY])
			# )
	elif 'AF' in list(dict_element.keys()):
		# if HARS is misaligned from the body...
		R = MAKE_E_MATRIX([('z',tconst.PSI_OFFSET),('y',tconst.THETA_OFFSET),('x',tconst.PHI_OFFSET)])
		# if config.debug:
			# print(
			# "rotation.MOUNT_TO_BODY_MATRIX, IBAL, returning R321(%.0f,%.0f,%.0f)"
			# %(tconst.PSI_OFFSET,tconst.THETA_OFFSET,tconst.ROLL_OFFSET)
			# )
	else:
		print("rotation.py: Unclear how attitude angles are defined, returning Identity")
		R = np.eye(3)
		
	return np.array(R)
    
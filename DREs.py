""" 
This is where the test specific data reduction equations (DREs) should be included.  Replace everything in the eval() 
function with the appropriate DREs to map test inputs/measurands (data, testinfo) to test results/VOIs.  eval() should accept 
data and testinfo as inputs and return the test results in the same format as the 'data' variable.  Additional functions used 
inside eval() may be defined in this file as appropriate.

"""

import re
import tconst
import numpy as np
import copy
import LSWT


def eval(data,G):

    D = LSWT.DREs(data,G)  # Replace lines in this function with your evaluation function that maps DRE inputs to results/VOIs.  Will replace this with a .exe eventually.

    return D


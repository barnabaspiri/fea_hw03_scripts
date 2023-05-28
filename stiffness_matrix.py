# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:53:02 2023

@author: barnabaspiri
"""

def Ke(I, E, L): # material stiffness
    
    import numpy as np
    
    Ke = I*E/L**3*np.matrix([[12,   6*L,     -12,   6*L],
                             [6*L,  4*L**2,  -6*L,  2*L**2],
                             [-12,  -6*L,    12,    -6*L],
                             [6*L,  2*L**2,  -6*L,  4*L**2]])
    
    return Ke

def KGe(N, E, L): # geometric stiffness

    import numpy as np
    
    # N: normal stress resultant
    KGe = N/(30*L)*np.matrix([[36,    3*L,     -36,    3*L],
                              [3*L,   4*L**2,  -3*L,  -L**2],
                              [-36,  -3*L,     36,   -3*L],
                              [3*L,  -L**2 ,  -3*L,   4*L**2]])
        
    return KGe

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:53:02 2023

@author: barnabaspiri
"""

def Me(rho, A, L): # consistent mass matrix
    
    import numpy as np
    
    Me = rho*A*L/420*np.matrix([[156,   22*L,     54,   -13*L],
                                [22*L,  4*L**2,  13*L,  -3*L**2],
                                [54,    13*L,    156,    -22*L],
                                [-13*L,  -3*L**2,  -22*L,  4*L**2]])
    
    return Me



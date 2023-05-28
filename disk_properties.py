# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:53:02 2023

@author: barnabaspiri
"""

def m_disk(D, d, t, RHO): # consistent mass matrix
    
    import numpy as np
    
    m_disk = (D**2-d**2)*np.pi/4*t*RHO
    
    return m_disk

def THETA_disk(D, d, t, m_disk): # consistent mass matrix
    
    THETA_disk = 1/4*m_disk*((D/2)**2+(d/2)**2) + 1/12*m_disk*t**2
    
    return THETA_disk

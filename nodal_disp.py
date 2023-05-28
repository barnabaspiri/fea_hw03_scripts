# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:08:17 2023

@author: barnabaspiri
"""

def eDOF(ecs):
    
    import numpy as np

    
    eDOF = np.matrix([2*ecs[0,0]-1, 2*ecs[0,0], 2*ecs[0,1]-1, 2*ecs[0,1]])
    
    #eDOF = np.zeros(3)
    
    return eDOF
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:03:23 2023

@author: barnabaspiri
"""

def ExtMatrix(mx, rows, size):
    
    import numpy as np
    
    n = rows.shape[1]
    Mx = np.zeros((size,size))
    
    for i in range(0,n):
        for j in range(0,n):

            Mx[rows[0,i]-1,rows[0,j]-1] = mx[i,j]
            
    return Mx

def SubMatrix(Mx, rows):
    
    
   import numpy as np
   
   n = rows.shape[1]
   mx = np.zeros((n,n))
   
   for i in range(0,n):
       for j in range(0,n):
           
           mx[i,j] = Mx[rows[0,i]-1,rows[0,j]-1]
    
   return mx
       
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:32:43 2023

@author: barnabaspiri
"""
#%%
from IPython import get_ipython
ipython = get_ipython()

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%%
# Setting the path to acces modules
import os
import sys

file_dir = os.path.dirname(__file__)

print(f"Working directory: {file_dir}")
sys.path.append(file_dir)

import numpy as np

# Geometric data
L_1 = 0.38 # m
D_1 = 0.2 # m
omega = 380 # rad/s
d = 0.048 # m
F_0 = 90 # N
M_0 = 7 # Nm

# Calculated geometric data
D_2 = 0.8*D_1
D_3 = 1.3*D_1
t_1 = D_1/10
t_2 = D_2/15
t_3 = D_3/12

# Material data
E = 210e9 # Pa
RHO = 7860 # kg/m^3

# DOF number of the system
DOF = 10

# Beam divided into 4 equal elements
# No. 1.: 0 - 0.38
# No. 2.: 0.38 - 0.76
# No. 3.: 0.76 - 1.14
# No. 4.: 1.14 - 1.52

Le = L_1
Ie = d**4*np.pi/64
Ae = d**2*np.pi/4
Ee = E
RHOe = RHO

ecs = np.matrix([[1,2],[2,3],[3,4],[4,5]]) # element - node table

#%% Matrix calculation
from stiffness_matrix import Ke # material stiffness matrix

K1 = Ke(Ie, Ee, Le)
K2 = Ke(Ie, Ee, Le)
K3 = Ke(Ie, Ee, Le)
K4 = Ke(Ie, Ee, Le)

from mass_matrix import Me # consistent mass matrix

M1 = Me(RHOe, Ae, Le)
M2 = Me(RHOe, Ae, Le)
M3 = Me(RHOe, Ae, Le)
M4 = Me(RHOe, Ae, Le)

from disk_properties import m_disk, THETA_disk # disk properties

m_disk_1 = m_disk(D_1, d, t_1, RHOe)
m_disk_2 = m_disk(D_2, d, t_2, RHOe)
m_disk_3 = m_disk(D_3, d, t_3, RHOe)

THETA_disk_1 = THETA_disk(D_1, d, t_1, m_disk_1)
THETA_disk_2 = THETA_disk(D_2, d, t_2, m_disk_2)
THETA_disk_3 = THETA_disk(D_3, d, t_3, m_disk_3)

M_disk = np.zeros([DOF,DOF])

M_disk[0,0] = m_disk_1
M_disk[1,1] = THETA_disk_1

M_disk[4,4] = m_disk_2
M_disk[5,5] = THETA_disk_2

M_disk[8,8] = m_disk_3
M_disk[9,9] = THETA_disk_3

#%% Matrix assemly 
from nodal_disp import eDOF

eDOF1 = eDOF(ecs[0])
eDOF2 = eDOF(ecs[1])
eDOF3 = eDOF(ecs[2])
eDOF4 = eDOF(ecs[3])

from glob_cond import ExtMatrix, SubMatrix

K = ExtMatrix(K1, eDOF1, DOF) + ExtMatrix(K2, eDOF2, DOF) + ExtMatrix(K3, eDOF3, DOF) + ExtMatrix(K4, eDOF4, DOF)
M = ExtMatrix(M1, eDOF1, DOF) + ExtMatrix(M2, eDOF2, DOF) + ExtMatrix(M3, eDOF3, DOF) + ExtMatrix(M4, eDOF4, DOF)
M = M + M_disk

freeDOF = np.matrix([1,2,4,5,6,8,9,10])

K_cond = SubMatrix(K, freeDOF)
M_cond = SubMatrix(M, freeDOF)

#%% Calculation of eigenfrequencies using numerical method
A = np.linalg.inv(M_cond) @ K_cond

(eigenVALS, eigenVECS) = np.linalg.eig(A)

eigenANGULARFREQS_sort = np.sort(np.sqrt(eigenVALS))

eigenFREQS_sort = np.sort(eigenANGULARFREQS_sort)/(2*np.pi)

print(f'\nThe natural frequencies:\n{eigenFREQS_sort}')

#%% First eigenfrequency and mode shape by Rayleigh's principle
import sympy as sp

x, a, b = sp.symbols("x, a, b")

Y = x**2 + a*x + b

eq1 = sp.Eq(Y.subs(x, L_1), 0)
eq2 = sp.Eq(Y.subs(x, 3*L_1), 0)

sol = sp.solve((eq1, eq2), (a,b))

a = round(sol[a], 4)
b = round(sol[b], 4)

print(f'\nThe first mode shape can be estimated as: Y(x) = x^2 + {a}*x + {b}')

Y = x**2 + a*x + b
Y_d = sp.diff(Y, x)
Y_dd = sp.diff(Y_d, x)

num = Ie*Ee*sp.integrate(Y_dd**2, (x, 0, 4*L_1))
den = RHOe*Ae*sp.integrate(Y, (x, 0, 4*L_1)) + m_disk_1*Y.subs(x,0)**2 + THETA_disk_1*Y_d.subs(x,0)**2 + m_disk_2*Y.subs(x,2*L_1)**2 + THETA_disk_2*Y_d.subs(x,2*L_1)**2 + m_disk_3*Y.subs(x,4*L_1)**2 + THETA_disk_3*Y_d.subs(x,4*L_1)**2

omega_RAY = np.sqrt(np.float64(num/den))
freq_RAY = omega_RAY/(2*np.pi)

print(f"\nThe first natural frequency based on Rayleigh's principle: {freq_RAY}")

#%% Direct time integration using Fox-Goodwin scheme
ALPHA = 1/12 # scheme parameters
GAMMA = 1/2

# Stability limit in terms of the 4th natural angular frequency: (omega_n8*dt < 2.45)

omega_n8 = eigenANGULARFREQS_sort[7]

T_n4 = 2*np.pi/omega_n8

dt = T_n4 / 3

stab_lim = omega_n8 * dt

print(f'\nChosen time step: {round(dt,8)} [s]')
print(f'\nStability limit omega_n4*dt = {round(stab_lim,4)}, if < 2.45, then OK.')

#%% Discrete time scale setup
n = 3000 # number of time steps
t = n*dt # time scale

U = np.zeros( (n+1, DOF-2, 1) )
U_d = np.zeros( (n+1, DOF-2, 1) )
U_dd = np.zeros( (n+1, DOF-2, 1) )

F = np.zeros( (n+1, DOF-2, 1) )
for i in range(n+1):
    F[i,2,0] = M_0*np.cos(omega*i*dt)

M = M_cond
K = K_cond

# import matplotlib.pyplot as plt
# t_scale = np.arange(0,t,dt)
# plt.plot(t_scale,F[:,2,0])
# plt.xlim(0,0.1)

#%% Initial conditions

# U = 0 and U_d = 0 at every timestep already established

U_dd[0] = np.linalg.inv(M) @ (F[0]-K@U[0])

#%% Running the numerical simulations

for i in range(n):
    
    U[i+1] = np.linalg.inv( M/(ALPHA*dt**2) + K) @ ( F[i] + M @ (U[i]/(ALPHA*dt**2) + U_d[i]/(ALPHA*dt) + (1/(2*ALPHA)-1)*U_dd[i]) )
    
    U_d[i+1] = GAMMA/(ALPHA*dt)*(U[i+1] - U[i]) + (1-GAMMA/ALPHA)*U_d[i] + (1-GAMMA/(2*ALPHA))*U_dd[i]*dt
    
    U_dd[i+1] = 1/(ALPHA*dt**2)*(U[i+1] - U[i]) - (1/(ALPHA*dt))*U_d[i] - (1/(2*ALPHA)-1)*U_dd[i]


#%% Plotting the displacement and the angle of rotation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

t_scale = np.arange(0,t+dt/2,dt)

# Displacement plot
plt.figure(figsize=(16/2.54, 10/2.54))
plt.plot(t_scale,U[:,3,0], linewidth=1)
plt.grid(True, alpha=0.5, linestyle='--')
# plt.xlim(0,0.14);
# plt.ylim(0,8000);
plt.xlabel('$\\mathrm{Time, t ~ (s)}$', labelpad=10, size=12);
plt.ylabel('$\\mathrm{Displacement ~ of ~ cross ~ section ~ K,} ~ v_3 ~ \\mathrm{(m)}$', labelpad=10, size=12);
plt.xticks(fontsize=12);
plt.yticks(fontsize=12);
plt.savefig("K_displacement.pdf",bbox_inches='tight',pad_inches=2/25.4)

# Angle of rotation plot
plt.figure(figsize=(16/2.54, 10/2.54))
plt.plot(t_scale,U[:,4,0], linewidth=1, color='#d95319')
plt.grid(True, alpha=0.5, linestyle='--')
# plt.xlim(0,0.14);
# plt.ylim(0,8000);
plt.xlabel('$\\mathrm{Time, t ~ (s)}$', labelpad=10, size=12);
plt.ylabel('$\\mathrm{Angle ~ of ~ cross ~ section ~ K,} ~ \\varphi_3 ~ \\mathrm{(rad)}$', labelpad=10, size=12);
plt.xticks(fontsize=12);
plt.yticks(fontsize=12);
plt.savefig("K_angle.pdf",bbox_inches='tight',pad_inches=2/25.4)


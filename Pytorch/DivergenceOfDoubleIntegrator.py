#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:09:40 2018

@author: lenard
"""
import matplotlib.pyplot as plt

import numpy as np

import control as ctrl


#%%
dt = 0.25
InputValue = 1.


#%%
# Continuous state-space matrices
A_cont = np.array([[0., 1.],[0., 0]])
B_cont = np.array([[0.],[1.]])
C_cont = np.array([[1., 0.],[0., 1.]])
D_cont = np.array([[0.],[0.]])

# Continuous system
SS_cont = ctrl.ss(A_cont, B_cont, C_cont, D_cont)
# Discrete system
SS_disc_nodelay = ctrl.sample_system(SS_cont, dt, method='tustin')


#%%
T = np.arange(0,100,dt)
yout, T, xout =ctrl.lsim(SS_disc_nodelay, U = np.ones(T.shape)*InputValue, T = T)

#%%
fig = plt.figure()
plt.plot(T, yout[0,:], label='Position')
plt.plot(T, yout[1,:], label='Velocity')
plt.plot(T, yout[1,:]/yout[0,:], label='Divergence')
plt.axhline(y=dt, color='black', linestyle='--', label='dt')
plt.ylim((-100,100))
plt.legend()
plt.grid()
plt.show()
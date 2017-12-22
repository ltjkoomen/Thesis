# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:33:35 2017

@author: Lenard
"""

import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt


#%% Functions
def MyFunction(w, p_branch):
    
    if w < p_branch:
        f = w**2
    else:
        f = np.sin(w)
        
    return f


#%% Calculations
# Point at which the function moves from a parabola to a sine wave
p_branch = 1.5

# Function that outputs the derivative w.r.t. w
GradFunction = grad(MyFunction,argnum=0)

# .. second derivative w.r.t. w
GradGradFunction = grad(GradFunction,argnum=0)

# Array of inputs
w = np.linspace(-2,5,num=1000)

# Arrays with function output, output first derivative, output second derivative
f, fdot, fdotdot = np.zeros((len(w))), np.zeros((len(w))), np.zeros((len(w)))
for k,dumw in enumerate(w):
    f[k]    = MyFunction(dumw, p_branch)
    fdot[k] = GradFunction(dumw, p_branch)
    fdotdot[k] = GradGradFunction(dumw, p_branch)


#%% Plot results
fig = plt.figure()
plt.plot(w, f, label='Output')
plt.plot(w, fdot, label='First derivative')
plt.plot(w, fdotdot, label='Second derivative')
plt.axvline(x=p_branch, label='Branching point', linestyle='--', color='black', linewidth=2)
plt.grid()
plt.legend()

plt.title('Automatic differentiation of a branched function')
plt.xlabel('Input parameter w')
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:33:35 2017

@author: Lenard
"""

import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt


#%% Functions
def MyFunction1(w, p_branch):
    
    if w < p_branch:
        f = w**2
    else:
        f = np.sin(w)
        
    return f

def MyFunction2(w):
    
    if w < p_branch:
        f = -1.
    else:
        f = 1.
        
    return f


#%% Calculations
# Point at which the function moves from a parabola to a sine wave
p_branch = 1.5

# Function that outputs the derivative w.r.t. w
GradFunction1 = grad(MyFunction1,argnum=0)

# .. second derivative w.r.t. w
GradGradFunction1 = grad(GradFunction1,argnum=0)

# Array of inputs
w = np.linspace(-2,5,num=1000)

# Arrays with function output, output first derivative, output second derivative
f1, fdot1, fdotdot1 = np.zeros((len(w))), np.zeros((len(w))), np.zeros((len(w)))
for k,dumw in enumerate(w):
    f1[k]    = MyFunction1(dumw, p_branch)
    fdot1[k] = GradFunction1(dumw, p_branch)
    fdotdot1[k] = GradGradFunction1(dumw, p_branch)


#%% Plot results
fig = plt.figure()
plt.plot(w, f1, label='Output')
plt.plot(w, fdot1, label='First derivative')
plt.plot(w, fdotdot1, label='Second derivative')
plt.axvline(x=p_branch, label='Branching point', linestyle='--', color='black', linewidth=2)
plt.grid()
plt.legend()

plt.title('Automatic differentiation of a branched function')
plt.xlabel('Input parameter w')
plt.show()

#%%
# Function that outputs the derivative w.r.t. w
GradFunction2 = grad(MyFunction2,argnum=0)

# .. second derivative w.r.t. w
GradGradFunction2 = grad(GradFunction2,argnum=0)

# Array of inputs
w = np.linspace(-2,5,num=1000)

# Arrays with function output, output first derivative, output second derivative
f2, fdot2, fdotdot2 = np.zeros((len(w))), np.zeros((len(w))), np.zeros((len(w)))
for k,dumw in enumerate(w):
    f2[k]    = MyFunction2(dumw)
    fdot2[k] = GradFunction2(dumw)
    fdotdot2[k] = GradGradFunction2(dumw)


#%% Plot results
fig = plt.figure()
plt.plot(w, f2, label='Output')
plt.plot(w, fdot2, label='First derivative', linewidth=2)
plt.plot(w, fdotdot2, label='Second derivative')
plt.axvline(x=p_branch, label='Branching point', linestyle='--', color='black', linewidth=2)
plt.grid()
plt.legend()

plt.title('Automatic differentiation of a branched function')
plt.xlabel('Input parameter w')
plt.show()
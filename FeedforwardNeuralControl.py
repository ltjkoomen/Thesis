# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:27:12 2017

@author: Lenard
"""
import autograd.numpy as np
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
from matplotlib import pyplot as plt

#%%
def InitParams(LayerSizes, rs=np.random):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net. Uses Xavier initialization"""
    return tuple([(np.sqrt(2/(m+n)) * rs.randn(n,m),   # weight matrix
             np.sqrt(0) * rs.randn(m))      # bias vector
            for m, n in zip(LayerSizes[1:],LayerSizes[:-1])])

def FeedforwardNet(Params, x):
    """Implements a fully connected feedforward neural network, with tanh activation function.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
       
#    for W,bias in Params[:-1]:       
#        x = np.dot(x, W) + bias
#        x = np.tanh(x)
    
    x = np.dot(x, Params[-1][0]) #+ Params[-1][1]

    return x

def StateIterationFun(Sys, x, y, NeuralInput, yTarget, NetParams):
      
#    NeuralInput = np.concatenate((np.array([yTarget[0]-y[0], y[0]]), np.array([yTarget[0]-y[0], y[0]]) - NeuralInput[0:2]))
    
    NeuralInput = np.array([yTarget[0]-y[0]])
    
    # Obtain control signal from neural net
    ControlSignal = FeedforwardNet(NetParams,NeuralInput)
    # State iteration
    x = np.dot(Sys['A'],x) + np.dot(Sys['B'],ControlSignal)
    # Output calculation
    y = np.array([x[1]/x[0]])
    
    return x,y,NeuralInput

def SimulationFun(Sys, NetParams, x0, yTarget, T, ErrorW, Bounds):
    
    N =  int(T/Sys['dt']) + 1
    
    x           = np.concatenate((np.array(x0), np.zeros((Sys['delay']*len(x0)))))
    y           = np.array([x[1]/x[0]])
    NeuralInput = np.zeros((LayerSizes[0]))
    yTargetUse  = np.array(yTarget)
    
    ErrorW = np.array(ErrorW)
    
    cost = 0.
    for k in range(N):
        x,y,NeuralInput  = StateIterationFun(Sys, x, y, NeuralInput, yTargetUse, NetParams)
        cost = cost + GetCost(y, yTargetUse, ErrorW)

        if not (Bounds[0][0] < x[0] < Bounds[0][1] and Bounds[1][0] < x[1] < Bounds[1][1]):
            N = k + 1
            break
        
    return x, cost/N

def SimulationSaveAllFun(Sys, NetParams, x0, yTarget, T, ErrorW):
    
    N =  int(T/Sys['dt']) + 1
    
    x           = np.concatenate((np.array(x0), np.zeros((Sys['delay']*len(x0)))))
    y           = np.array([x[1]/x[0]])
    NeuralInput = np.zeros((LayerSizes[0]))
    yTargetUse  = np.array(yTarget)
    
    ErrorW = np.array(ErrorW)
    
    xTotal    = np.zeros((N, len(x)))
    yTotal    = np.zeros((N, len(y)))
    costTotal = np.zeros(N)
    
    xTotal[0,:]  = x
    yTotal[0,:]  = y
    costTotal[0] = GetCost(y, yTargetUse, ErrorW)
    
    cost = 0.
    for k in range(1,N):
        x,y,NeuralInput  = StateIterationFun(Sys, x, y, NeuralInput, yTargetUse, NetParams)
        cost = cost + GetCost(y, yTargetUse, ErrorW)
        
        xTotal[k,:]  = x
        yTotal[k,:]  = y
        costTotal[k] = cost/N
        
    return xTotal, yTotal, costTotal

def GetCost(y, yTarget, w):
    Diff = yTarget - y
    Cost = np.dot(np.square(Diff), w)
    return Cost

def ObjectiveFunWrap(NetParams, k):
    TotalCost = 0
    Count = 0
    for yTargetUse in yTarget:
        for x0Use in x0:
            _,DumTotalCost = SimulationFun(Sys, NetParams, x0Use, yTargetUse, T, ErrorW, Bounds)
            TotalCost = TotalCost + DumTotalCost
            Count = Count + 1
    return (TotalCost/Count)

def PrintPerf(Params, iter, _):
    if iter == 0:
        print("     Epoch     |    Train cost  ")
    if iter%5 == 0:
        Cost = ObjectiveFunWrap(Params, iter)
        print(str(iter) + '  ' + str(np.round(Cost,6)))
        
def GetZoomLimits(x, sparefactor):
    boundaries = np.array([x.min(), x.max()])
    span = boundaries[1] - boundaries[0]
    boundaries_stretched = boundaries + span*np.array([-sparefactor, sparefactor])
    return tuple(boundaries_stretched)

def InitiateDelayedinputSystem(Sys, NumDelay):
    
    if NumDelay == 0: return Sys
    
    A,B,C,D,dt = Sys['A'], Sys['B'], Sys['C'], Sys['D'], Sys['dt']
    
    NumStates = A.shape[0]
    
    ADelay = np.zeros((NumStates+NumDelay*B.shape[0], NumStates+NumDelay*B.shape[0]))
    ADelay[0:NumStates,0:NumStates] = A
    for k in range(NumStates):
        ADelay[k, NumStates + k*NumDelay] = 1
    
    DelayVector = np.ones(NumDelay*NumStates)
    DelayVector[(NumDelay-1):NumDelay*NumStates:NumDelay] = 0
    ADelay[NumStates:, NumStates:] = np.diag(DelayVector[:-1],1) 
    
    BDelay = np.zeros((ADelay.shape[0],1))
    for k in range(NumStates):
        BDelay[NumStates+(k+1)*NumDelay-1,0] = B[k,0]
    
    CDelay = np.zeros((C.shape[0],ADelay.shape[0]))
    CDelay[0:C.shape[0], 0:C.shape[1]] = C
    
    DDelay = np.zeros((D.shape[0],ADelay.shape[0]))
    DDelay[0:D.shape[0], 0:D.shape[1]] = D
    
    SysDelay = {'A':ADelay, 'B':BDelay, 'C':CDelay, 'D':DDelay, 'dt':dt, 'delay':NumDelay}
    
    return SysDelay


#%% INITIALIZATION
# Number of neurons per layer
LayerSizes = (1,1,1)

# Randomly define a certain amount of starting states between certain bounds
NumInitializationsX = 1
InitializationBoundsX = ((100., 100.),(-0.0, 0.))

x0 = tuple([(float(np.random.rand(1))*(InitializationBoundsX[0][1]-InitializationBoundsX[0][0]) + InitializationBoundsX[0][0], 
             float(np.random.rand(1))*(InitializationBoundsX[1][1]-InitializationBoundsX[1][0]) + InitializationBoundsX[1][0]) for k in range(NumInitializationsX)])

# Define a certain number of target outputs
yTarget = tuple([(k,) for k in np.linspace(-0.1, -0.33, 1)])

# Error weighting
ErrorW = (1.,)

# State bounds, when out of bounds the simulation stops
Bounds = ((0.05, 100),(-np.inf, np.inf))

# Time parameters
dt = 0.1
T  = 100

# Initialize discrete state-space system (double integrator)
A = np.array([[1., dt],[0., 1.]])
B = np.array([[0.5*dt**2],[dt]])
C = np.eye(2)
D = np.array([[0.],[0.]])

SysRaw = {'A':A, 'B':B, 'C':C, 'D':D, 'dt':dt, 'delay':0}

# Define equivalent state-space system with delayed input
# Number of samples delay
NumDelay = 0
# SS system
Sys = InitiateDelayedinputSystem(SysRaw, NumDelay)

# %%
# Initialize neural net parameters
ParamsInitial = InitParams(LayerSizes)

# Flatten parameter tuple
ParamsInitialFlat, UnflattenParams = flatten(ParamsInitial)

# Get gradient of objective using autograd.
ObjectiveGrad = grad(ObjectiveFunWrap,argnum=0)


#%% Train neural net
# Adam optimizer
# The optimizers provided can optimize lists, tuples, or dicts of parameters.
#ParamsOpt = adam(CostGrad, Params, step_size=StepSize, num_iters=NumEpochs, callback=PrintPerf)
ParamsOpt = adam(ObjectiveGrad, ParamsInitial, callback=PrintPerf, num_iters=100,
         step_size=0.01, b1=0.9, b2=0.99, eps=10**-8)


#%%

x0      = (10., 0.)
yTarget = (-0.1,)
T       = 300.

SpareRoomFactor = 0.05

xPlot, yPlot, costPlot = SimulationSaveAllFun(Sys, ParamsOpt, x0, yTarget, T, ErrorW)

divPlot = xPlot[:,1]/xPlot[:,0]

tPlot  = np.arange(0.,T+Sys['dt'],Sys['dt'])
tPlot  = tPlot[0:len(costPlot)]

try:
    IdxCollision = np.where(xPlot[:,0] < 0)[0][0]
except:
    IdxCollision = len(tPlot)

plt.figure()
plt.subplot(2,1,1)
plt.plot(tPlot, xPlot[:,0], label='Height')
plt.plot(tPlot, xPlot[:,1], label='Velocity')

plt.xlim(GetZoomLimits(tPlot[0:IdxCollision+1], SpareRoomFactor))
plt.ylim(GetZoomLimits(xPlot[0:IdxCollision+1,:], SpareRoomFactor))
plt.grid()
plt.title('Simulation output')
#plt.xlabel('Time [s]')
plt.ylabel('Simulation output')
plt.legend()
#plt.xlim((0, 35))
plt.show()

plt.subplot(2,1,2)
plt.plot(tPlot, divPlot, label='Divergence')
plt.axhline(y=yTarget[0], label='Reference value', linestyle='--', color='black', linewidth=2)
plt.xlim(GetZoomLimits(tPlot[0:IdxCollision+1], SpareRoomFactor))
#plt.ylim((-0.05, 0.05))
#plt.xlim((0, 35))
#plt.ylim((0, 35))
#plt.ylim(GetZoomLimits(divPlot[0:IdxCollision+1-10], SpareRoomFactor))
plt.xlabel('Time [s]')
plt.legend()
plt.grid()

plt.show()


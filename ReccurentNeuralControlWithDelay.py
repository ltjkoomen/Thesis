# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:28:45 2017

@author: Lenard
"""

import autograd.numpy as np
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
from matplotlib import pyplot as plt



#%%

def SimpleRNN(Params, u, x):
    """Implements a first-order recurrent neural network, with tanh activation function.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    
    x = np.dot(x, Params[1][0]) + np.dot(u, Params[0][0]) + Params[1][1]
    x = np.tanh(x)
    
    y = np.dot(x, Params[-1][0]) + Params[-1][1]
    
    return y,x


def InitParams(LayerSizes, rs=np.random.RandomState(1)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net. Uses Xavier initialization"""
    
    LayerIdx = [0, 1, 1, 2]
    
    Params = [[np.sqrt(2/(LayerSizes[n]+LayerSizes[m])) * rs.randn(LayerSizes[n],LayerSizes[m]),   # weight matrix
             np.sqrt(0) * rs.randn(LayerSizes[m])]      # bias vector
            for n, m in zip(LayerIdx[:-1],LayerIdx[1:])]
    
    # Remove bias weights from input layer since it is not needed
    Params[0][1] = 0.
    
    ParamsTuple = tuple([tuple([Params[i][0],Params[i][1]])for i in range(len(LayerSizes))])
    
    return ParamsTuple


def StateIterationFun(Sys, x, y, NeuralState, NeuralInput, ControlSignal, yTarget, NetParams):
    
    NeuralInput = np.concatenate((np.array([yTarget[0]-y[0], y[0]]), np.array([yTarget[0]-y[0], y[0]]) - NeuralInput[0:2], ControlSignal))
    
    # Obtain control signal from neural net
    ControlSignal,NeuralState = SimpleRNN(NetParams,NeuralInput,NeuralState)
    # State iteration
    x = np.dot(Sys['A'],x) + np.dot(Sys['B'],ControlSignal)
    # Output calculation
    y = np.array([x[1]/x[0]])
    
    return x,y,NeuralState,NeuralInput,ControlSignal


def SimulationFun(Sys, NetParams, x0, yTarget, T, ErrorW, Bounds):
    
    N =  int(T/Sys['dt']) + 1
    
    x             = np.concatenate((np.array(x0), np.zeros((Sys['delay']*len(x0)))))
    y             = np.array([x[1]/x[0]])
    NeuralState   = np.zeros((LayerSizes[1]))
    NeuralInput   = np.zeros((LayerSizes[0]))
    ControlSignal = np.zeros((LayerSizes[-1]))
    yTargetUse    = np.array(yTarget)
    
    ErrorW = np.array(ErrorW)
    
    cost = 0.
    for k in range(N):
        if k < 1:
            yTargetUse = np.array([0.])
        else:
            yTargetUse  = np.array(yTarget)
        
        x,y,NeuralState,NeuralInput,ControlSignal  = StateIterationFun(Sys, x, y, NeuralState, NeuralInput, ControlSignal, yTargetUse, NetParams)
        cost = cost + GetCost(y, yTargetUse, ErrorW)
        
        if not (Bounds[0][0] < x[0] < Bounds[0][1] and Bounds[1][0] < x[1] < Bounds[1][1]):
            N = k + 1
            break
    
    return x, cost/N


def SimulationSaveAllFun(Sys, NetParams, x0, yTarget, T, ErrorW):
    # Simulates the landing and saves the entire state and outputs etc.
    N =  int(T/Sys['dt']) + 1
    
    x             = np.concatenate((np.array(x0), np.zeros((Sys['delay']*len(x0)))))
    y             = np.array([x[1]/x[0]])
    NeuralState   = np.zeros((LayerSizes[1]))
    NeuralInput   = np.zeros((LayerSizes[0]))
    ControlSignal = np.zeros((LayerSizes[-1]))
    yTargetUse    = np.array(yTarget)
    
    ErrorW = np.array(ErrorW)
    
    xTotal    = np.zeros((N, len(x)))
    yTotal    = np.zeros((N, len(y)))
    costTotal = np.zeros(N)
    
    xTotal[0,:]  = x
    yTotal[0,:]  = y
    costTotal[0] = GetCost(y, yTargetUse, ErrorW)
    
    cost = 0.
    for k in range(1,N):
        if k < 0:
            yTargetUse = np.array([0.])
        else:
            yTargetUse  = np.array(yTarget)
        
        x,y,NeuralState,NeuralInput,ControlSignal  = StateIterationFun(Sys, x, y, NeuralState, NeuralInput, ControlSignal, yTargetUse, NetParams)
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
    return TotalCost/Count

def PrintPerf(Params, iter, _):
    if iter == 0:
        print("     Epoch     |    Train cost  ")
    if iter%5 == 0:
        Cost = ObjectiveFunWrap(Params, iter)
        Gradient = flatten(ObjectiveGrad(Params, iter))
        print(str(iter) + '  ' + str(np.round(Cost,6)) + '  ' + str(np.square(Gradient[0]).sum()))


def GetZoomLimits(x, sparefactor):
    # Finds the lmits of a vector so the plots can be zoomed in properly
    boundaries = np.array([x.min(), x.max()])
    span = boundaries[1] - boundaries[0]
    boundaries_stretched = boundaries + span*np.array([-sparefactor, sparefactor])
    return tuple(boundaries_stretched)


def InitiateDelayedinputSystem(Sys, NumDelay):
    # Creates a SS system equivalent to the second order system, but with the 
    #difference that the inputs enter a sequency of delays before affecting the state
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


#%%
# Number of neurons per layer
LayerSizes = (5,15,1)

# Initial state
#PositionStartVector = (1, 10, 50, 100)
#x0 = tuple([(k, 0) for k in PositionStartVector])

# Randomly define a certain amount of starting states between certain bounds
NumInitializationsX = 1
InitializationBoundsX = ((10., 10.),(-0., 0.))

x0 = tuple([(float(np.random.rand(1))*(InitializationBoundsX[0][1]-InitializationBoundsX[0][0]) + InitializationBoundsX[0][0], 
             float(np.random.rand(1))*(InitializationBoundsX[1][1]-InitializationBoundsX[1][0]) + InitializationBoundsX[1][0]) for k in range(NumInitializationsX)])

# Define a certain number of target outputs
yTarget = tuple([(k,) for k in np.linspace(-0.25, -0.33, 1)])

# Error weighting
ErrorW = (1.,)

# State bounds, when out of bounds the simulation stops
Bounds = ((0.05, 1000),(-np.inf, np.inf))

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

#%%
# Initialize neural net parameters
ParamsInitial = InitParams(LayerSizes)

# Flatten parameter tuple
ParamsInitialFlat, UnflattenParams = flatten(ParamsInitial)

# Get gradient of objective using autograd.
ObjectiveGrad = grad(ObjectiveFunWrap,argnum=0)

#%%
# The optimizers provided can optimize lists, tuples, or dicts of parameters.
#ParamsOpt = adam(CostGrad, Params, step_size=StepSize, num_iters=NumEpochs, callback=PrintPerf)
ParamsOpt = adam(ObjectiveGrad, ParamsInitial, callback=PrintPerf, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8)

#%%

x0      = (10, 0.)
yTarget = (-0.25,)
T       = 100.

# Increases extra room in the zoom of the plot
SpareRoomFactor = 0.1

# Simulate the landing using the optimized parameters and chosen target and initial state
xPlot, yPlot, costPlot = SimulationSaveAllFun(Sys, ParamsOpt, x0, yTarget, T, ErrorW)
# Calculate divergence 
divPlot = xPlot[:,1]/xPlot[:,0]
# Define time vector
tPlot  = np.arange(0.,T+Sys['dt'],Sys['dt'])
tPlot  = tPlot[0:len(costPlot)]

# Find when the drone collides with the ground
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
plt.xlabel('Time [s]')
plt.ylabel('Simulation output')
plt.legend()
plt.show()

plt.subplot(2,1,2)
plt.plot(tPlot, divPlot, label='Divergence')
plt.xlim(GetZoomLimits(tPlot[0:IdxCollision+1], SpareRoomFactor))
plt.ylim(GetZoomLimits(divPlot[0:IdxCollision+1-10], SpareRoomFactor))

plt.title('')
plt.xlabel('Time [s]')
plt.ylabel('Divergence')
plt.legend()
plt.grid()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:00:35 2018

@author: lenard


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

import control as ctrl

#%%
def weight_init(m):
    
    classname = m.__class__.__name__
    
    print(m)
    
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
        
    return m

def ss_init(dt,delay=0,method='tustin'):
    
    # Continuous state-space matrices
    A_cont = np.array([[0., 1.],[0., 0]])
    B_cont = np.array([[0.],[1.]])
    C_cont = np.array([[1., 0.],[0., 1.]])
    D_cont = np.array([[0.],[0.]])
    
    numState = A_cont.shape[0]
    
    # Continuous system
    SS_cont = ctrl.ss(A_cont, B_cont, C_cont, D_cont)
    # Discrete system
    SS_disc_nodelay = ctrl.sample_system(SS_cont, dt, method=method)
    
    if delay == 0: return SS_disc_nodelay
    
    # Initialize discrete delayed state-space matrices
    dumA = np.zeros((numState*(delay+1), numState*(delay+1)))
    dumB = np.zeros((numState*(delay+1), 1))
    dumC = np.zeros((SS_disc_nodelay.C.shape[0],numState*(delay+1)))
    dumD = np.zeros((SS_disc_nodelay.D.shape[0],SS_disc_nodelay.B.shape[1]))
    dumvec = np.ones((numState*delay))
    
    # Populate ss-matrices such that the input is delayed by delay samples
    dumA[0:numState, 0:numState] = SS_disc_nodelay.A
    for k in range(numState):
        dumA[k, numState+k*delay] = 1.
        dumB[numState-1+(k+1)*delay, 0] = SS_disc_nodelay.B[k]
        dumvec[(k+1)*delay-1] = 0.
        
    dumvec = dumvec[:-1]
    
    dumA[numState:,numState:] = np.diag(dumvec, k=1)
    
    dumC[:SS_disc_nodelay.C.shape[0],:SS_disc_nodelay.C.shape[1]] = SS_disc_nodelay.C
    
    dumD[:SS_disc_nodelay.D.shape[0],:SS_disc_nodelay.D.shape[1]] = SS_disc_nodelay.D
        
    
    # Create delayed discrete state-space system
    SS_disc_delay = ctrl.ss(dumA, dumB, dumC, dumD, dt)
    
    return SS_disc_delay
    

class Simulation(nn.Module):

    def __init__(self):
        super(Simulation,self).__init__()
        
        #% Simulation parameters
        self.SimulationParameters = SimulationParameters
        
        #% State space layers
        self.SS = {}
        self.SS['A']  = nn.Linear(StateSpaceSystem.A.shape[1], StateSpaceSystem.A.shape[0], bias=False)
        self.SS['B']  = nn.Linear(StateSpaceSystem.B.shape[1], StateSpaceSystem.B.shape[0], bias=False)
        self.SS['C']  = nn.Linear(StateSpaceSystem.C.shape[1], StateSpaceSystem.C.shape[0], bias=False)
        self.SS['D']  = nn.Linear(StateSpaceSystem.D.shape[1], StateSpaceSystem.D.shape[0], bias=False)
        
        self.SS['dt']    = StateSpaceSystem.dt
        self.SS['delay'] = delay
            
        #% Controller layers
        self.fc1 = nn.Linear(1,20, bias=True)
        self.fc2 = nn.Linear(20,1, bias=True)
     
    def GetNeuralInput(self, state, inputs):
        netinput  = torch.cat((inputs[0] - state[1]/state[0]))
        netinput  = netinput.view(1,-1)
        return netinput
        
    def forward(self, inputs):
              
        ss_state  = self.SimulationParameters['x0'] #Initial state
          
        outputs = Variable(torch.zeros(self.SimulationParameters['N_steps'], \
                                       self.SimulationParameters['N_batch'], \
                                       self.SimulationParameters['N_sim_output']))
        
        for k in range(self.SimulationParameters['N_steps']):

            #---------------------Neural controller----------------------------
            # Get input from current state
            net_input = self.GetNeuralInput(ss_state, inputs[k,:,:])

            # Feedforward neural net 
            net_output = self.fc1(net_input)
#            net_output = F.tanh(net_output)
#            net_output = self.fc2(net_output)

            #---------------------State-space model----------------------------
            # State iteration
            ss_state = self.SS['A'](ss_state) + self.SS['B'](net_output)
            # Output
            output = ss_state[1]/ss_state[0]
            
            #----------------------Final output--------------------------------
            # Define forward output
            outputs[k,:,:] = output
            
            #----------------------Monitoring output---------------------------
            outputs_extra[k,:,:] = np.concatenate((ss_state.data.numpy()), axis=2)
            
            
            #----------------------Collision check-----------------------------
            if ss_state[0].data.numpy() < self.SimulationParameters['StoppingBounds'][0] or \
               ss_state[0].data.numpy() > self.SimulationParameters['StoppingBounds'][1]: return outputs[:k+1]
        
        return outputs, outputs_extra
    
def CreateInput(N_steps, N_batch, RefArray):
    
    RefArray = np.reshape(RefArray, (1, -1))
    
    if RefArray.size == 1:
        RefArray = RefArray.repeat(N_batch, axis=1)
    elif RefArray.size != N_batch:
        print('Number of reference inputs does not match number of batches')
        return
    
    InputArray = np.ones((N_steps, N_batch, 1)) * RefArray[:,:,None]
    
    return InputArray

def InitializeStatespaceLayers(net, StateSpaceSystem):
    
    net.SS['A'].weight.data = torch.from_numpy(StateSpaceSystem.A).float()
    net.SS['B'].weight.data = torch.from_numpy(StateSpaceSystem.B).float()
    net.SS['C'].weight.data = torch.from_numpy(StateSpaceSystem.C).float()
    net.SS['D'].weight.data = torch.from_numpy(StateSpaceSystem.D).float()
    
    net.SS['A'].weight.requires_grad = False
    net.SS['B'].weight.requires_grad = False
    net.SS['C'].weight.requires_grad = False
    net.SS['D'].weight.requires_grad = False
    
    return net
    
                
#%%
# Time vector
dt          = 0.05
T           = 50.

N_steps     = int(T/dt) + 1
TimeVector  = np.arange(0., T+dt, dt)

# State-space model of drone
delay       = 0 #Input delay
dmethod     = 'tustin'

StateSpaceSystem = ss_init(dt, delay, dmethod)

addstates = int(StateSpaceSystem.A.shape[0] - 2) #States to add due to input delay
ss_state0 = Variable(torch.cat((torch.from_numpy(np.array([100., -0.0])).float(), torch.zeros(addstates).float())))

#%%
N_batch = 10

# Divergence reference and target
RefArray  = np.array([-0.2])

# input = (N_steps, N_batch, N_input), target = (N_steps, N_batch, N_output)
DumInput  = CreateInput(N_steps, N_batch, RefArray)
DumTarget = DumInput

Input  = Variable(torch.from_numpy(DumInput).float())
Target = Variable(torch.from_numpy(DumTarget).float())


#%% Simulation parameters
SimulationParameters = {}
SimulationParameters['dt']             = dt
SimulationParameters['ss_state0']      = ss_state0
SimulationParameters['StoppingBounds'] = (0.05, 1000)
SimulationParameters['N_steps']        = N_steps
SimulationParameters['N_batch']        = N_batch
SimulationParameters['N_sim_output']   = 1



#%%
# Initialize net
net = Simulation()

# Initialize control layer weights
net.apply(weight_init)

# Initialize state space layer weights
net = InitializeStatespaceLayers(net, StateSpaceSystem)


#%%
# Output without training
OutputStart = net(Input)


#%% Optimization criterion and algorithm
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


#%%
NumEpochs = 200

for epoch in range(1):  # loop over the dataset multiple times
    for i in range(NumEpochs):
        
        # Zero the gradient to disconnect from previous epoch
        optimizer.zero_grad()
        
        # Forward
        Output,UsedSteps = net(Input)
        loss = criterion(Output[:UsedSteps+1], Target[:UsedSteps+1])
        
        # Backward
        loss.backward()
        
        # Optimize
        optimizer.step()

        # print statistics
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.data.numpy()))

print('Finished Training')

Output,_ = net(Input)

#%%

fig = plt.figure()
plt.plot(TimeVector, Target.data.numpy(), label='Target', color='k' )
plt.plot(TimeVector[:OutputStart.data.numpy().shape[0]], OutputStart.data.numpy(), label='Before training' )
plt.plot(TimeVector[:Output.data.numpy().shape[0]], Output.data.numpy(), label='After training' )
plt.legend()
plt.grid()
plt.show()
 
#%%
#T, yout = ctrl.impulse_response(StateSpaceSystem, TimeVector, np.zeros(StateSpaceSystem.A.shape[0]))
#
#fig = plt.figure()
#plt.plot(T, yout[:,0], label='height')
#plt.plot(T, yout[:,1], label='velocity')
#plt.legend()
#plt.show()
#




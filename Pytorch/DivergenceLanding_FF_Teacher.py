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
    print(classname)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
#        m.bias.data.fill_(0)
        
    elif classname.find('RNN') != -1:
        m.weight_hh_l0.data.normal_(0.0, 0.1)
        m.weight_ih_l0.data.normal_(0.0, 0.1)
        m.bias_hh_l0.data.fill_(0)
        m.bias_ih_l0.data.fill_(0)

        
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
        
        # Neural network
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1,1, bias=False)
        # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(50,1)
                
        # State space system
        self.SS = {}
        self.SS['A']  = Variable(torch.from_numpy(StateSpaceSystem.A).float(), requires_grad=False)
        self.SS['B']  = Variable(torch.from_numpy(StateSpaceSystem.B).float(), requires_grad=False)
        self.SS['C']  = Variable(torch.from_numpy(StateSpaceSystem.C).float(), requires_grad=False)
        self.SS['D']  = Variable(torch.from_numpy(StateSpaceSystem.D).float(), requires_grad=False)
        self.SS['dt'] = StateSpaceSystem.dt
        self.SS['delay'] = delay
        
        # Simulation parameters
        self.SimulationParameters = {}
        self.SimulationParameters['N']  = N_steps
        if self.SS['delay'] !=0:
            addstates = int(self.SS['A'].data.numpy().shape[0] - ss_state0.data.numpy().shape[0])
            self.SimulationParameters['x0'] = torch.cat((ss_state0, Variable(torch.zeros((addstates)))))
        else:
            self.SimulationParameters['x0'] = ss_state0
            
        self.SimulationParameters['StoppingBounds'] = (0.05, 1000)
    
    def GetNeuralInput(self, state, inputs):
        netinput  = torch.cat((inputs[0] - state[1]/state[0]))
        netinput  = netinput.view(1,-1)
        return netinput
        
    def forward(self, inputs):
        
        TeacherForce = False
        
        ss_state  = self.SimulationParameters['x0'].unsqueeze(1) # Initial state
          
        outputs = Variable(torch.zeros(N_steps))
        for k in range(N_steps):
            
            if k%1==0:
                if np.random.rand(1) > 0.5:
                    TeacherForce = True
                else:
                    TeacherForce = False
                
            if TeacherForce:
                ss_state = Variable(torch.from_numpy(np.array([ss_state[0].data.numpy(), ss_state[0].data.numpy()*inputs[k].data.numpy()])).float())
            
            #---------------------Neural controller----------------------------
            # Get input from current state
            net_input = self.GetNeuralInput(ss_state, inputs[k])

            # Feedforward neural net 
            net_output = self.fc1(net_input)
#            net_output = F.tanh(net_output)
#            net_output = self.fc2(net_output)

            #---------------------State-space model----------------------------
            # State iteration
            ss_state  = torch.mm(self.SS['A'], ss_state) + torch.mm(self.SS['B'], net_output)
            # Output
            ss_output = ss_state[1]/ss_state[0]
            
            #----------------------Final output--------------------------------
            # Define forward output
            outputs[k] = ss_output
            
            #----------------------Collision check-----------------------------
            if ss_state[0].data.numpy() < self.SimulationParameters['StoppingBounds'][0] or \
               ss_state[0].data.numpy() > self.SimulationParameters['StoppingBounds'][1]: return outputs[:k+1], k
        
        return outputs, k


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
delay   = 0
dt      = 0.05
T       = 50.
N_steps = int(T/dt) + 1
TimeVector  = np.arange(0., T+dt, dt)

dmethod = 'tustin'

ss_state0 = Variable(torch.from_numpy(np.array([100., -0.0])).float())

D_ref = -0.2
#DumTarget = -(1/(TimeVector+1)-1) * D_ref
DumTarget = np.ones(TimeVector.shape) * D_ref
 
# Get state-space model
StateSpaceSystem = ss_init(dt, delay, dmethod)
    
#%%
# Initialize net
net = Simulation()
# Initialize weighdt
net.apply(weight_init)

print(net)

#%%
# Input = (batchsize, input..)
Input         = Variable(torch.from_numpy(DumTarget).float())
Target        = Input
OutputStart,_ = net(Input)


#%%
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#optimizer = optim.SGD(net.parameters(), lr=0.005)

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




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:34:46 2018

@author: lenard
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
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
#        m.bias.data.fill_(0)
        
    elif classname.find('RNN') != -1:
        m.weight_hh_l0.data.normal_(0.0, 0.001)
        m.weight_ih_l0.data.normal_(0.0, 0.001)
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
    
    # Populate ss-matrices such that the input is delayed by #delay samples
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
 
        # Delay buffer layer
        self.DelayBuffer_x = nn.Linear(6, 6, bias=False)
        self.DelayBuffer_u = nn.Linear(1, 6, bias=False)
    
        
        # RNN
        self.rnn = nn.RNN(input_size=12, hidden_size=20, num_layers=1, nonlinearity='tanh', bias=True)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(12,20, bias=True)
        self.fc2 = nn.Linear(20,1)
        
    
    def forward(self, inputs, targets, epoch, maxepoch, forceswitch):
        
        ForceLevel = epoch/(0.5 * maxepoch)
        
        rnn_state       = None
        outputs         = Variable(torch.zeros(targets.shape).float(), requires_grad=False)
        
        output = Variable(torch.zeros((1,inputs.shape[1],1)).float(), requires_grad=False)
        
        delay_buffer = Variable(torch.zeros(1,inputs.shape[1],6).float(), requires_grad=False)
        
        force_sum = 0
        for k in range(inputs.shape[0]):
           
#            ForceLevel = 0.0001 * 1.01**k
            
            if forceswitch > 0:
                if k > 0:
                    if np.random.rand(1) > ForceLevel or forceswitch==2:
                        force_sum += 1
                        output = targets[[k-1],:,:]
            
            delay_buffer = self.DelayBuffer_x(delay_buffer) + self.DelayBuffer_u(output[:,:,[0]])
            
            neural_input = torch.cat((delay_buffer, inputs[[k],:,:]), dim=2)
            
#            _,rnn_state = self.rnn(neural_input, rnn_state)
            
            output = self.fc1(neural_input)
            rnn_state = F.tanh(output)
            
            output = self.fc2(rnn_state)
        
            outputs[k] = output
        
#        print(force_sum)

        return outputs
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
delay   = 0
dt      = 0.05
T       = 10.
N_steps = int(T/dt) + 1
TimeVector  = np.arange(0., T+dt, dt)

dmethod = 'tustin'

# Get state-space model
StateSpaceSystem = ss_init(dt, delay, dmethod)
    
#%%
# Initialize net
net = Simulation()
# Initialize weighdt
net.apply(weight_init)
# Make delay buffer layer
net.DelayBuffer_x.weight.data = torch.from_numpy(np.eye(6, k=1)).float()
net.DelayBuffer_x.weight.requires_grad = False

dum = np.zeros((6,1))
dum[-1] = 1.
net.DelayBuffer_u.weight.data = torch.from_numpy(dum).float()
net.DelayBuffer_u.weight.requires_grad = False

print(net)

#%%
# Input = (batchsize, input..)
NumTimesteps = TimeVector.shape[0]
NumInputs    = 1
NumDelayedElements = (5,)
NumBatches   = 100
NumOutputs   = 2
#Input        = Variable(torch.rand(NumTimesteps, NumBatches, NumInputs).float()-0.5)
#Input        = Variable(torch.from_numpy(0.1*np.sin(5*np.repeat(TimeVector.reshape(-1,1),NumBatches,axis=1))).float()).view(NumTimesteps,NumBatches,NumInputs)

Input = np.zeros((NumTimesteps, NumBatches, int((np.array(NumDelayedElements)+1).sum())))
for i in range(NumInputs):
    for j in range(NumBatches):
        dumA = np.random.rand(1) * 1
        dumF = 2 + 1 * np.random.rand(1) * 2 * np.pi
        dumP = np.random.rand(1) * 2 * np.pi
        duminput = dumA * np.sin(TimeVector*dumF + dumP) + np.random.randn(TimeVector.shape[0])*0.1
        
#        duminput = (np.random.rand(TimeVector.shape[0])-0.5)
        
        duminput_delay = np.zeros((duminput.shape[0], NumDelayedElements[i]+1))
        for k in range(NumDelayedElements[i]+1):
            duminput_delay[k:,k] = np.roll(duminput, k)[k:]
        
        idxbounds = (int((np.array(NumDelayedElements[:i])+1).sum()), int((np.array(NumDelayedElements[:i+1])+1).sum()))
        
        Input[:,j,idxbounds[0]:idxbounds[1]] = duminput_delay
        
        
Input = Variable(torch.from_numpy(Input).float(), requires_grad=False)

Target = np.zeros((NumTimesteps, NumBatches, NumOutputs))
k = 0
while k < NumBatches :
    dumX0 = (np.random.rand(2)-np.array([0.5, 0.5]))*np.array([10., 0]) + np.array([10., 0])
    dumtarget,_,_ = ctrl.lsim(StateSpaceSystem, Input[:,k,0].data.numpy(), TimeVector, X0=dumX0)
    
    Target[:,k,0] = dumtarget[0,:]
    Target[:,k,1] = dumtarget[1,:]
    k+= 1
    
    

Target = Target[:,:,[1]] / Target[:,:,[0]]  
Target = Target[:,:,[0]]

#%%
NumEpochs = 2000

Target        = Variable(torch.from_numpy(Target).float(), requires_grad=False)
OutputStart   = net(Input, Target, NumEpochs, NumEpochs, 0)


#%%
criterion = nn.MSELoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


#%%


NumBatchUse = 20

NumHidden = 10

for i in range(NumEpochs):
    
    # Zero the gradient to disconnect from previous epoch
    optimizer.zero_grad()
    
    # Pick batches
    if NumBatches > 1:
        IdxBatch = Variable(torch.LongTensor(np.random.randint(low=0, high=NumBatches-1, size=NumBatchUse)))
    else:
        IdxBatch = Variable(torch.LongTensor([0]))
    
    # Forward
    Output = net(torch.index_select(Input,1,IdxBatch), torch.index_select(Target,1,IdxBatch), i, NumEpochs, 2)
    loss   = criterion(Output, torch.index_select(Target,1,IdxBatch))
    
    # Backward
    loss.backward()
    
    # Optimize
    optimizer.step()

    # print statistics
    if i % 10 == 9:    # print every 10 mini-batches
        print('[%5d] loss: %.3f' %
              (i + 1, loss.data.numpy()))

print('Finished Training')

Output = net(Input, Target, 0, NumEpochs, 2)


#%%
for k in range(NumBatches):

    fig = plt.figure()
    plt.plot(TimeVector, Target[:,k,0].data.numpy(), label='Target', color='k' )
    plt.plot(TimeVector, OutputStart[:,0,0].data.numpy(), label='Before training' )
    plt.plot(TimeVector, Output[:,k,0].data.numpy(), label='After training' )
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

plt.figure()
plt.plot(Input[:,0,0].data.numpy())
plt.show()




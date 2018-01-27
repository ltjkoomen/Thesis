#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 00:16:00 2017

@author: lenard
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

#%%
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
        
    elif classname.find('RNN') != -1:
        m.weight_hh_l0.data.normal_(0.0, 0.1)
        m.weight_ih_l0.data.normal_(0.0, 0.1)
        m.bias_hh_l0.data.fill_(0)
        m.bias_ih_l0.data.fill_(0)
        
    return m
          
class RecurrentNet(nn.Module):

    def __init__(self):
        super(RecurrentNet,self).__init__()
        
        # RNN
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, nonlinearity='tanh', bias=True, dropout=0.0)
        # an affine operation: y = Wx + b
        self.h20 = nn.Linear(50,1)
        
    def forward(self, inputs):
          
        hidden = None # Allow RNN to initialize
        
        # Recurrent neural net 
        output, hidden = self.rnn(inputs.view(-1,1,self.rnn.input_size), hidden)
        
        # Linear map to final output
        output = self.h20(output.squeeze(1))
        
        return output, hidden

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
# Initialize net
net = RecurrentNet()
# Initialize weights
net.apply(weights_init)

#%%
print(net)

# Input = (batchsize, inputs..)
Input  = Variable(torch.linspace(-20,20,1000))
Target = Variable(torch.np.sin(Input.data))
OutputStart, _ = net(Input)

#%%
criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


#%%
NumEpochs = 1000



for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(NumEpochs):
        
        # Zero the gradient to disconnect from previous epoch
        optimizer.zero_grad()
        
        # Forward
        Output,_  = net(Input)
        loss      = criterion(Output, Target)
        
        # Backward
        loss.backward()
        
        # Optimize
        optimizer.step()

        # print statistics
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.data.numpy()))

print('Finished Training')

Output,_  = net(Input)

#%%
fig = plt.figure()
plt.scatter(Input.data.numpy(), Target.data.numpy(), label='Target', color='k' )
plt.scatter(Input.data.numpy(), OutputStart.data.numpy(), label='Before training' )
plt.scatter(Input.data.numpy(), Output.data.numpy(), label='After training' )
plt.legend()
plt.show()
 






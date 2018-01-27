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
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)
            
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
          
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1, 25)
        self.fc2 = nn.Linear(25,1)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
# Initialize net
net = Net()
# Initialize weights
net.apply(weights_init)

print(net)

# Input = (batchsize, inputs..)
Input  = Variable(4*(torch.rand(100, 1)-0.5))
Target = Variable(torch.np.sin(Input.data.sum(dim=1)))
OutputStart = net(Input)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


#%%
NumEpochs = 20000



for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(NumEpochs):
        
        # Zero the gradient to disconnect from previous epoch
        optimizer.zero_grad()
        
        # Forward
        Output  = net(Input)
        loss    = criterion(Output, Target)
        
        # Backward
        loss.backward()
        
        # Optimize
        optimizer.step()

        # print statistics
        if i % 100 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.data.numpy()))

print('Finished Training')

Output  = net(Input)

#%%
fig = plt.figure()
plt.scatter(Input.data.numpy(), Target.data.numpy(), label='Target', color='k' )
plt.scatter(Input.data.numpy(), OutputStart.data.numpy(), label='Before training' )
plt.scatter(Input.data.numpy(), Output.data.numpy(), label='After training' )
plt.legend()
plt.show()
 






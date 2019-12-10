import torch
import torchvision
import torch.nn as nn
import numpy as np 
import torchvision.transforms as transforms

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

#create tensor
x = torch.tensor(1.,requires_grad = True)
w = torch.tensor(2.,requires_grad = True)
b = torch.tensor(3.,requires_grad = True)

#build a computational graph.
y = w*x + b

#compute gradients
y.backward()

#print out the gradients
print(x.grad)  #x.grad = 2
print(w.grad)  #w.grad = 1
print(b.grad)  #b.grad = 1

# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

#create tensor of shape (10,3) and (10,2)
x = torch.randn(10,3)
y = torch.randn(10,2)

#build a fully connected layer.
linear = nn.Linear(3,2)
print('w: ',linear.weight)
print('b: ',linear.bias)

#build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(),lr = 0.01)

#forward pass.
pred = linear(x)

#compute loss
loss = criterion(pred,y)
print('loss: ',loss.item())

#backward pass
loss.backward()

#print out the gradients
print('dL/dw: ',linear.weight.grad)
print('dL/db: ',linear.bias.grad)

#1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

#print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred,x)
print('loss after 1 step optimization: ',loss.item())
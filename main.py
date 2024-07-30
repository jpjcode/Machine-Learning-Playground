import torch
import torch.nn as nn
import torch.optim as optim

from LNN import LiquidStateMachine



#creates the model
model = LiquidStateMachine(1, 1, [8])

#gets optimizer an criterion(mean the cost function)
opt = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

#forward for output
output = model.forward(torch.ones(1, 1))

#This is not a good way to test LSM in my opinion but it is a start
for i in range(10000):
    opt.zero_grad()  #clears gradients
    output = model.forward(torch.ones(1, 1))  #forwards the model like setting an input
    target = torch.ones(1, 1)  #here is the bad testing I was talking about
    loss = criterion(output, target)  #gets the loss from cost function
    print(loss)

    loss.backward()
    opt.step()

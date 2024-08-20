import torch
import torch.nn as nn
import torch.optim as optim

from LNN import LiquidStateMachine



#creates the model
model = LiquidStateMachine(1, 1, [64])

#gets optimizer and criterion(the cost function)
opt = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

#forward for output
output = model.forward(torch.ones(1, 1))

#This is not a good way to test LSM in my opinion but it is a start
for i in range(1000):
    opt.zero_grad()  #clears gradients
    output = model.forward(torch.ones(1, 1))
    target = torch.ones(1, 1) * i #I notice the spiking neurons are able to keep up with changing output inspite of fixed input
    loss = criterion(output, target)
    print(loss)

    loss.backward()
    opt.step()

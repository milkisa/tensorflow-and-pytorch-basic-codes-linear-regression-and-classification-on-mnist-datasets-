import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
x= torch.tensor([-1.0, 0.0,1.0,2.0,3.0,4.0])
x= x.cuda()
y= torch.tensor([-3.0, -1.0, 1.0, 3.0,5.0,7.0])
y= y.view(-1,1)
print(y)
y= y.cuda()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.fc2 = nn.Linear(1,1)

    def forward(self,x):

        x = self.fc2(x)
        return x
device= ("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)
criterion=  nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr= 0.01)
training_loss= []
for epoch in range(500):
    optimizer.zero_grad()
    z= model(x.view(-1,1))
    loss= criterion(z, y)
    loss.backward()
    optimizer.step()
    training_loss.append(loss)
    print("training loss at {} is {}".format(epoch,loss))
a= model(torch.tensor([[10.0]]).cuda())
print(a)
plt.plot(training_loss, label='training_loss')
plt.legend()
plt.show()



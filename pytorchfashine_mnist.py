import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transforms = transforms.Compose([transforms.Resize(16),transforms.ToTensor(), transforms.Normalize((0.1307,),(0.1307,))])
train_set= datasets.FashionMNIST('FashionMNIST/', download=True,train= True, transform= transforms)
validation_set= datasets.FashionMNIST('FashionMNIST/', download= True, train= False, transform= transforms)
trainloader= torch.utils.data.DataLoader(train_set, batch_size=4, shuffle = True)
validationLoader= torch.utils.data.DataLoader(validation_set, batch_size= 4, shuffle= True)
device= ("cuda" if torch.cuda.is_available() else "cpu")
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.conv1= nn.Conv2d(1,128,3)
        self.maxpool1= nn.MaxPool2d(2)
        self.conv2= nn.Conv2d(128,128,3)
        self.maxpool2=nn.MaxPool2d(2)
        self.drop= nn.Dropout(0.2)
        self.linear1= nn.Linear(512,128)
        self.linear2= nn.Linear(128,10)
    def forward(self,x):
        x= self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x= self.drop(x)
        x= x.view(-1,512)


        x= self.linear1(F.relu(x))
        x= self.linear2(x)
        return x
model= NET()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters())
total_training_loss= []
total_validating_loss= []
total= 0
train_accuracy= []
validat_accuracy=[]
for epochs in range(10):
    total= 0
    total_train_loss=0
    total_val_loss=0
    model.train()
    for idx, (image, label) in enumerate(trainloader):
        image, label= image.to(device),label.to(device)
        optimizer.zero_grad()
        pred= model(image)
        loss= criterion(pred, label)
        total_train_loss+=loss
        loss.backward()
        optimizer.step()
        _, yhat= torch.max(pred,1)
        total+=(yhat==label).sum().item()
    accuracy= total/len(train_set)
    train_accuracy.append(accuracy)
    total_train_loss= total_train_loss/(idx+1)
    total_training_loss.append(total_train_loss)
    total= 0
    model.eval()
    for idx ,(image, label) in enumerate(validationLoader):
        image, label = image.to(device) , label.to(device)
        optimizer.zero_grad()
        pred= model(image)
        loss = criterion(pred, label)
        total_val_loss+= loss

        _, yhat= torch.max(pred,1)
        total+=(yhat==label).sum().item()
    accuracy_val= total/len(validation_set)
    validat_accuracy.append(accuracy_val)
    total_vali= total_val_loss/(idx+1)
    total_validating_loss.append(total_vali)
    model.train()
    print("Epoch: {}/{}  ".format(epochs, 10),
              "Training loss: {:.4f}  ".format(total_train_loss),
              "Testing loss: {:.4f}  ".format(total_vali),
              "Train accuracy: {:.4f}  ".format(accuracy),
              "Test accuracy: {:.4f}  ".format(accuracy_val))
import matplotlib.pyplot as plt
epoch= range(len(train_accuracy))
plt.plot(epoch,train_accuracy, label= 'training accuracy')
plt.plot(epoch, validat_accuracy, label= "validation accuracy")
plt.plot(epoch, total_training_loss, label= "trainin  loss")
plt.plot(epoch, total_val_loss, label = "validation loss")
plt.legend()
plt.show()









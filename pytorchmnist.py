import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))  ])
train_set = datasets.MNIST('DATA_MNIST/', download=True, train=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

validation_set = datasets.MNIST('DATA_MNIST/', download=True, train=False, transform=transform)
validationLoader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
import torch.nn as nn
import torch.nn.functional as F
# training_data = enumerate(trainloader)
# batch_idx, (images, labels) = next(training_data)
# print(type(images)) # Checking the datatype
# print(images.shape) # the size of the image
# print(labels.shape) # the size of the labels




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        # [(28 + 2*0 - 3)/1] + 1 = 26
        self.pool = nn.MaxPool2d(kernel_size=2)
        # (26/2,26/2,32)
        # (13*13*32= 5408)
        self.fc1 = nn.Linear(5408, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = x.view(-1, 5408 )
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
import time

start_time = time.time()

epochs =5
train_loss, val_loss = [], []
accuracy_total_train, accuracy_total_val = [], []


for epoch in range(epochs):

    total_train_loss = 0
    total_val_loss = 0



    total = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):

        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()

        pred = model(image)

        loss = criterion(pred, label)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # pred = torch.nn.functional.softmax(pred, dim=1)

        _,yhat= torch.max(pred.data, 1)
        total+=(yhat==label).sum().item()

    accuracy_train = total / len(train_set)
    accuracy_total_train.append(accuracy_train)

    total_train_loss = total_train_loss / (idx + 1)
    train_loss.append(total_train_loss)

    # validating our model

    total = 0
    for idx, (image, label) in enumerate(validationLoader):
        image, label = image.cuda(), label.cuda()
        # print(label.shape)
        pred = model(image)
        loss = criterion(pred, label)
        total_val_loss += loss.item()

        pred = torch.nn.functional.softmax(pred, dim=1)
        _,yhat= torch.max(pred.data, 1)
        total+=(yhat==label).sum().item()

    accuracy_val = total / len(validation_set)
    accuracy_total_val.append(accuracy_val)

    total_val_loss = total_val_loss / (idx + 1)
    val_loss.append(total_val_loss)


    print("Epoch: {}/{}  ".format(epoch, epochs),
              "Training loss: {:.4f}  ".format(total_train_loss),
              "Testing loss: {:.4f}  ".format(total_val_loss),
              "Train accuracy: {:.4f}  ".format(accuracy_train),
              "Test accuracy: {:.4f}  ".format(accuracy_val))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Test loss')
plt.legend()
plt.grid()
# img = images[0]
# img = img.to(device)
# img = img.view(-1, 1, 28, 28)
# print(img.shape)
#
# # Since we want to use the already pretrained weights to make some prediction
# # we are turning off the gradients
# with torch.no_grad():
#     logits = model.forward(img)
#
# probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
#
# print(probabilities)
#
# fig, (ax1, ax2) = plt.subplots(figsize=(6,8), ncols=2)
# ax1.imshow(img.view(1, 28, 28).detach().cpu().numpy().squeeze(), cmap='inferno')
# ax1.axis('off')
# ax2.barh(np.arange(10), probabilities, color='r' )
# ax2.set_aspect(0.1)
# ax2.set_yticks(np.arange(10))
# ax2.set_yticklabels(np.arange(10))
# ax2.set_title('Class Probability')
# ax2.set_xlim(0, 1.1)
#
# plt.tight_layout()


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt

from google.colab import drive
drive.mount('/content/drive/')

data_dir = 'drive/MyDrive/ml/Lab10'
classes = os.listdir(data_dir + '/Training')
print(classes)

train_dataset = ImageFolder(data_dir + "/Training", transform=tt.ToTensor())
print(train_dataset)
test_dataset = ImageFolder(data_dir + "/Test", transform=tt.ToTensor())
print(test_dataset)

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
for images, labels in dataloader:
    print(labels[0])
    print(train_dataset.class_to_idx)
    print(train_dataset.classes[labels[0]])
    plt.imshow(images[0].permute(1, 2, 0))
    break

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        ######################################################################
        #### DESIGN LAYERS :
        ### SEQUENCE: CONV1,ACTIVATION1,POOLING1,  CONV2,ACTIVATION2,POOLING2, LINEAR(FC)
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(4)
        self.layer3 = nn.Linear(32 * 11 * 11, len(classes))

    def forward(self, x):
        # COMBINE LAYERS
        ## 1) CONV1
        out = self.layer1(x)

        ## 2) ACTIVATION1
        out = self.relu(out)

        ## 3) POOLING1
        out = self.pool1(out)

        ## 4) CONV2
        out = self.layer2(out)

        ## 5) ACTIVATION2
        out = self.relu(out)

        ## 6) POOLING2
        out = self.pool2(out)

        ## 7) flatten ########## DURING LAB WE JUST FORGOT FOLLOWING FLATTEN LAYER ###############
        out = out.view(out.size(0), -1)

        ## 8) LINEAR(FC)
        return self.layer3(out)

# batch_size, epoch and iteration
batch_size = 100
num_epochs = (len(train_dataset.samples) / batch_size)
num_epochs = int(num_epochs)

# data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create CNN
model = CNNModel()

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for i, (images, labels) in enumerate(train_loader):

    train = images

    # Clear gradients
    optimizer.zero_grad()

    # Forward propagation
    outputs = model(train)

    # Calculate softmax and ross entropy loss
    loss = error(outputs, labels)

    # Calculating gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    count += 1

    if count % 10 == 0:
        # Calculate Accuracy
        correct = 0
        total = 0
        # Iterate through test dataset
        for images, labels in test_loader:
            test = images

            # Forward propagation
            outputs = model(test)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]

            # Total number of labels
            total += len(labels)

            correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        # store loss and iteration
        loss_list.append(loss.data)
        iteration_list.append(count)
        accuracy_list.append(accuracy)
        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
        if count == 500:
            break

# visualization loss
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()
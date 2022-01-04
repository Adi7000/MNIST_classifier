import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist1')
###################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 11
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
train_list = list((train_loader))
test_list = list((test_loader))

# add ~6000 randomly generated noise images to train on (6 images added to 938 batches)
# assume that generated images are noisy and probablitiy of image being recognizable is 0
for batch_pointer in range(len(train_loader)):
    train_list[batch_pointer][1] = torch.cat((train_list[batch_pointer][1], torch.tensor([10]*6)),0) # adding 10 (negative class) to the labels
    train_list[batch_pointer][0] = torch.cat((train_list[batch_pointer][0], torch.rand(6,1,28,28)),0) # add negative class images

# add ~1000 images to test data
for batch_pointer in range(len(test_loader)):
    test_list[batch_pointer][1] = torch.cat((test_list[batch_pointer][1], torch.tensor([10]*6)),0) # adding 10 (negative class) to the labels
    test_list[batch_pointer][0] = torch.cat((test_list[batch_pointer][0], torch.rand(6,1,28,28)),0) # add negative class images

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# Use to import a model
#model = torch.load("model_11.pth")
#model.eval()

# Loss and optimizer initialization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
      
      
# Train the model 
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
num_step = 0
num_samples = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_list):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        
        num_samples += predicted.size(0)
        num_step += 1
        if (i+1) % 100 == 0 or i == n_total_steps-1:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            ############## TENSORBOARD ########################
            writer.add_scalar('training loss', running_loss / num_step, epoch * n_total_steps + i)
            running_accuracy = running_correct / num_samples
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            num_step = 0
            num_samples = 0
            ###################################################


# generate graphs of model parameters
with torch.no_grad():
    parameters = list(model.parameters())
    weights1 = parameters[0]
    weights2 = parameters[2]        #parameters 1 holds the biases for layer 1
    
    # using superposition to construct class images 
    class_imgs = torch.matmul(weights2, weights1)    
    
    # layer 1 images
    ''' #for printing layer 1 images upto 500 images, resize subplot
    images = weights1.reshape(len(parameters[0]),28,28)
    images = images.numpy()
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(images[i], cmap='gray')
    plt.show()
    '''
    
    # showing superposition images
    class_imgs2 = class_imgs.reshape(num_classes,28,28)
    for i in range(num_classes):
        plt.subplot(6, 2, i+1)
        plt.imshow(class_imgs2[i], cmap='gray')
        plt.title(i)
    plt.show()      


# Testing, don't need to compute gradients (for memory efficiency)

class_labels = []
class_preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_list:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(predicted)

    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 11000 test images: {acc:.4f} %')

# save the model
torch.save(model, f'model_11.pth')

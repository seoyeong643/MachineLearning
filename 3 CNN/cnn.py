# import torch and other necessary modules from torch
import torch
import torch.nn as nn
import torch.optim as optim
...
# import torchvision and other necessary modules from torchvision 
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

...
from sklearn.metrics import accuracy_score, recall_score, precision_score

torch.manual_seed(42)

# recommended preprocessing steps: resize to square -> convert to tensor -> normalize the image
# if you are resizing, 100 is a good choice otherwise GradeScope will time out
# you could use Compose (https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) from transforms module to handle preprocessing more conveniently
transform = transforms.Compose([
    transforms.Resize((100, 100)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


# thanks to torchvision, this is a convenient way to read images from folders directly without writing datasets class yourself (you should know what datasets class is as mentioned in the documentation)
dataset = datasets.ImageFolder(root="./petimages", transform=transform)


# now we need to split the data into training set and evaluation set 
testSize = int(0.2 * len(dataset))
trainSize = len(dataset) - testSize
# use 20% of the dataset as test
test_set, train_set = torch.utils.data.random_split(dataset, [testSize, trainSize])
# prepare dataloader for training set and evaluation set
trainloader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)

# model hyperparameter
learning_rate = 0.01
batch_size = 50
epoch_size = 10



# model design goes here
class CNN(nn.Module):

    # there is no "correct" CNN model architecture for this lab, you can start with a naive model as follows:
    # convolution -> relu -> pool -> convolution -> relu -> pool -> convolution -> relu -> pool -> linear -> relu -> linear -> relu -> linear
    # you can try increasing number of convolution layers or try totally different model design
    # convolution: nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    # pool: nn.MaxPool2d (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    # linear: nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    def __init__(self):
        super(CNN,self).__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.c3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output layer, assuming 2 classes (Cats and Dogs)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.c1(x)))
        x = self.pool(nn.functional.relu(self.c2(x)))
        x = self.pool(nn.functional.relu(self.c3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU
# search in official website for CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)


# start model training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size): # begin with trying 10 epochs 

    running_loss = 0.0 # you can print out average loss per batch every certain batches

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, labels = data
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs=cnn(inputs)
        #Loss
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # print some statistics
        running_loss +=  loss.item()    # add loss for current batch 
        if i % 100 == 99:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() # turn on evaluation model, also a good practice to do
with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs, so turn on no_grad mode
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels.tolist() # convert labels to list and append to ground_truth
        # calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        # the class with the highest logit is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        prediction += predicted.tolist() # convert predicted to list and append to prediction


# GradeScope is chekcing for these three variables, you can use sklearn to calculate the scores
accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction)
precision = precision_score(ground_truth, prediction)







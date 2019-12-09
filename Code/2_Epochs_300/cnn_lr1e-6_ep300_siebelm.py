#!/usr/bin/env python
# coding: utf-8




# Import packages
import os
import glob
import numpy as np   
import random
import PIL.Image as pilimg
from keras.preprocessing.image import load_img
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

saveddata = "cnn_lr1e-6_ep300_siebelm.pt"






# Collect file names
train = 'train/'
test = 'test/'

x_train = []
for file in os.listdir(train):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        x_train.append(train + filename)
y_train = []
for file in os.listdir(train):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        y_train.append(train + filename)

x_test = []
for file in os.listdir(test):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        x_test.append(test + filename)
y_test = []
for file in os.listdir(test):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        y_test.append(test + filename)




# Length of datasets
print("Train:", len(y_train))
print("Test:", len(y_test))





# Set up target variable for training
labels = []
for file in y_train:
    label = open(file, "r")
    label = list(label.read().split('\n'))
    if label[0] == "1 Game of Thrones":
        label[0] = "1 Game of Thrones"
    else:
        label[0] = "0 Sitcom"
    labels.append(label[0])

# Label encoder
le = LabelEncoder()
le.fit(["0 Sitcom", "1 Game of Thrones"])

# One-hot encode data
y_train = le.transform(labels)
print(le.classes_)




print("Target values pre-oversampling")
print(np.unique(y_train, return_counts=True))





# Oversample GoT
x_ovsp = []
y_ovsp = []
for im, lab in zip(x_train, y_train):
    if lab == 1:
        x_ovsp.append(im)
        y_ovsp.append(lab)
        
y_ovsp = np.array(y_ovsp)

def shuffle_train(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

x_train = np.concatenate((x_train, x_ovsp), axis=0)
y_train = np.concatenate((y_train, y_ovsp), axis=0)
x_train, y_train = shuffle_train(x_train, y_train)




print("Target values post-oversampling")
print(np.unique(y_train, return_counts=True))





# Move Tensors to GPU (cuda) where possible
dtype = torch.float
if torch.cuda.is_available(): 
    device = torch.device("cuda:0")
else: 
    device = torch.device("cpu")
# Random seeds for PyTorch
torch.manual_seed(42)  
np.random.seed(42)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





# Data augmentation on photos
resize = 50

tf_train = transforms.Compose([
        transforms.RandomResizedCrop(resize),
        transforms.ColorJitter(.3, .3, .3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

tf_test = transforms.Compose([
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

images = []
for file in x_train:
    image = load_img(file, color_mode='rgb')
    image = tf_train(image)
    image = np.array(image)
    image = image.reshape(1,3,resize,resize)
    image = torch.FloatTensor(image).to(device)
    images.append(image)
    
x_train = images

images = []
for file in x_test:
    image = load_img(file, color_mode='rgb')
    image = tf_train(image)
    image = np.array(image)
    image = image.reshape(1,3,resize,resize)
    image = torch.FloatTensor(image).to(device)
    images.append(image)
    
x_test = images



# Hyper Parameters
batch_size = 256
learning_rate = 1e-6
n_epoch = 300
a_size = 1





# Data Loader (Input Pipeline)
x_train = torch.utils.data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
y_train = torch.utils.data.DataLoader(dataset=y_train, batch_size=batch_size, shuffle=True)





# Convolutional Neural Network Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        # Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)        
        self.convnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # Layer 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)        
        self.convnorm4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # Layer 5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2)        
        self.convnorm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # Output Layer
        self.linear1 = nn.Linear(256, 128) # ((50*(2^5))^2*256
        self.linear1_bn = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, a_size)
        # Activation
        self.act = torch.relu
        # Dropout
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x)))) 
        x = self.pool2(self.convnorm2(self.act(self.conv2(x)))) 
        x = self.drop(self.pool3(self.convnorm3(self.act(self.conv3(x)))))
        x = self.pool4(self.convnorm4(self.act(self.conv4(x))))        
        x = self.drop(self.pool5(self.convnorm5(self.act(self.conv5(x)))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x.view(len(x), -1))
        return torch.sigmoid(x)

    
model = CNN().to(device)

# Model optimization
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
bce_criterion = nn.BCELoss()





# Training function
def train_model(x_train, y_train, n_epoch):
    i = 1 
    for epoch in range(n_epoch):

        # Print column headers for each epoch
        print("") 
        print(
f" \
BCE batch loss; \
BCE running loss \
"
        )
        print(
"Epoch %i \
; Epoch %i" % (epoch + 1, epoch + 1)
        ) 
        
        # Prepare training
        model.train()
        running_loss = 0.0
        for batch, (inputs, target) in enumerate(zip(x_train, y_train)): 

            # Resize inputs
            x_tensor = inputs.reshape(inputs.shape[0], 3, resize, resize)
            # Set target as float
            y_tensor = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            
            # Drop gradients
            optimizer.zero_grad()
            
            # Forward propagation
            logits = model(x_tensor)
            
            # Performance index
            bce_loss = bce_criterion(logits, y_tensor)
            
            # Back propagation; update
            bce_loss.backward()
            optimizer.step()

            # Statistics
            running_loss += bce_loss.item()
            print(
f" \
{bce_loss.item():0.3f}; \
{running_loss / len(x_train):0.3f} \
"
            )

        # Save model
        torch.save(model.state_dict(), saveddata)





# Run training
train_model(x_train, y_train, n_epoch)        





# Load model
model.load_state_dict(torch.load(saveddata))
model.eval()
print("")
print(model)





# Prediction function
def predict(x):
    
    dtype = torch.float
    device = torch.device("cpu")
    
    # Random seeds for PyTorch
    torch.manual_seed(21)  
    np.random.seed(21)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data augmentation on photos
    resize = 50
    
    tf = transforms.Compose([
            transforms.RandomResizedCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    images = []
    length = len(x)
    for i in range(length):
        path = x[i]
        image = load_img(file, color_mode='rgb')
        image = tf(image)
        image = np.array(image)
        image = image.reshape(1,3,resize,resize)
        image = torch.FloatTensor(image).to(device)
        images.append(image)

    x = images
    
    # DataLoader
    x = torch.utils.data.DataLoader(dataset=x)

    # Output size
    a_size = 1
    
    # Convolutional Neural Network Architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # Layer 1
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
            self.convnorm1 = nn.BatchNorm2d(16)
            self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=2)
            # Layer 2
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
            self.convnorm2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            # Layer 3
            self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)        
            self.convnorm3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            # Layer 4
            self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)        
            self.convnorm4 = nn.BatchNorm2d(128)
            self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            # Layer 5
            self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2)        
            self.convnorm5 = nn.BatchNorm2d(256)
            self.pool5 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            # Output Layer
            self.linear1 = nn.Linear(256, 128) # ((50*(2^5))^2*256
            self.linear1_bn = nn.BatchNorm1d(128)
            self.linear2 = nn.Linear(128, a_size)
            # Activation
            self.act = torch.relu
            # Dropout
            self.drop = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.pool1(self.convnorm1(self.act(self.conv1(x)))) 
            x = self.pool2(self.convnorm2(self.act(self.conv2(x)))) 
            x = self.drop(self.pool3(self.convnorm3(self.act(self.conv3(x)))))
            x = self.pool4(self.convnorm4(self.act(self.conv4(x))))        
            x = self.drop(self.pool5(self.convnorm5(self.act(self.conv5(x)))))
            x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
            x = self.linear2(x.view(len(x), -1))
            return torch.sigmoid(x)

    model = CNN().to(device)
    model.load_state_dict(torch.load(saveddata))
    model.eval()
    
    # Initiate y_pred
    y_pred = torch.zeros(1, 1)
    y_logits = torch.zeros(1, 1)
    with torch.no_grad():
        for batch, inputs in enumerate(x):
        
            # Resize inputs
            x_tensor = inputs.reshape(inputs.shape[0]*inputs.shape[1], 3, resize, resize)
            
            # Model
            logits = model(x_tensor)
            
            # Add raw logits
            y_logits = torch.cat([y_logits, logits])
            
            # Hardlimit
            logits = np.array(logits)
            threshold = np.array([.5])
            np.putmask(logits, logits >= threshold, 1)
            np.putmask(logits, logits <  threshold, 0)
            y_lab = torch.FloatTensor(logits).to(device)
            
            # Add to y_pred
            y_pred = torch.cat([y_pred, y_lab])
    
    # Remove initiation row
    y_logits = y_logits[1:len(y_logits)]
    y_pred = y_pred[1:len(y_pred)]
    return y_pred, y_logits

y_pred, y_logits = predict(x_test)
y_pred = np.array(y_pred)
y_logits = np.array(y_logits)





# Set up target variable for testing
labels = []
for file in y_test:
    label = open(file, "r")
    label = list(label.read().split('\n'))
    if label[0] == "1 Game of Thrones":
        label[0] = "1 Game of Thrones"
    else:
        label[0] = "0 Sitcom"
    labels.append(label[0])

# Label encoder
le = LabelEncoder()
le.fit(["0 Sitcom", "1 Game of Thrones"])

# One-hot encode data
y_true = le.transform(labels)
print(le.classes_)





# Evaluation
print("Prediction distribution")
print(np.count_nonzero(y_pred == 0), np.count_nonzero(y_pred == 1))
print("True distribution")
print(np.count_nonzero(y_true == 0), np.count_nonzero(y_true == 1))
print(classification_report(y_true, y_pred))
print("")
print('Accuracy: {:1.4f}'.format(accuracy_score(y_true, y_pred)))
print('F1: {:1.4f}'.format(f1_score(y_true, y_pred)))
print('Precision: {:1.4f}'.format(precision_score(y_true, y_pred)))
print('Recall: {:1.4f}'.format(recall_score(y_true, y_pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


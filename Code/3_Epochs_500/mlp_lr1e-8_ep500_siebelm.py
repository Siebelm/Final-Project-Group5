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

saveddata = "mlp_lr1e-8_ep500_siebelm.pt"



# Collect file names
train = 'train/'
val = 'val/'

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

x_val = []
for file in os.listdir(val):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        x_val.append(val + filename)
y_val = []
for file in os.listdir(val):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        y_val.append(val + filename)



# Length of datasets
print("Train:", len(y_train))
print("Val:", len(y_val))





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
np.unique(y_train, return_counts=True)






# Oversample difficult
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
np.unique(y_train, return_counts=True)





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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
])

images = []
for file in x_train:
    image = load_img(file, color_mode='grayscale')
    image = tf_train(image)
    image = np.array(image)
    image = image.reshape(1,1,resize,resize)
    image = torch.FloatTensor(image).to(device)
    images.append(image)
    
x_train = images






# Hyper Parameters
batch_size = 256
learning_rate = 1e-8
n_epoch = 500
R = resize*resize 
S = 2500 
a_size = 1





# Data Loader (Input Pipeline)
x_train = torch.utils.data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
y_train = torch.utils.data.DataLoader(dataset=y_train, batch_size=batch_size, shuffle=True)





# Multi-layer Perceptron Neural Network Architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Layer 1
        self.linear1 = nn.Linear(R, S)
        self.linear1_bn = nn.BatchNorm1d(S)
        # Layer 2
        self.linear2 = nn.Linear(S, S)
        self.linear2_bn = nn.BatchNorm1d(S)
        # Layer 3
        self.linear3 = nn.Linear(S, S)        
        self.linear3_bn = nn.BatchNorm1d(S)
        # Layer 4
        self.linear4 = nn.Linear(S, S)        
        self.linear4_bn = nn.BatchNorm1d(S)
        # Layer 5
        self.linear5 = nn.Linear(S, S)        
        self.linear5_bn = nn.BatchNorm1d(S)
        # Output Layer
        self.linearO = nn.Linear(S, a_size)
        # Activation
        self.act = torch.relu
        # Dropout
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.linear1_bn(self.act(self.linear1(x)))
        x = self.drop(self.linear2_bn(self.act(self.linear2(x))))
        x = self.drop(self.linear3_bn(self.act(self.linear3(x))))
        x = self.drop(self.linear4_bn(self.act(self.linear4(x))))        
        x = self.drop(self.linear5_bn(self.act(self.linear5(x))))
        x = self.drop(self.linearO(x))        
        return torch.sigmoid(x)

    
model = MLP().to(device)

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
            x_tensor = inputs.view(inputs.shape[0], resize*resize)
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
    
    # Move Tensors to GPU (cuda) where possible
    dtype = torch.float
    device = torch.device("cpu")
    
    # Random seeds for PyTorch
    torch.manual_seed(42)  
    np.random.seed(42)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data augmentation on photos
    resize = 50
    
    tf = transforms.Compose([
            transforms.RandomResizedCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    images = []
    length = len(x)
    for i in range(length):
        path = x[i]
        image = load_img(file, color_mode='grayscale')
        image = tf(image)
        image = np.array(image)
        image = image.reshape(1,1,resize,resize)
        image = torch.FloatTensor(image).to(device)
        images.append(image)

    x = images
    
    # DataLoader
    x = torch.utils.data.DataLoader(dataset=x)

    # Output size
    a_size = 1
    
    # Multi-layer Perceptron Neural Network Architecture
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            # Layer 1
            self.linear1 = nn.Linear(resize*resize, S)
            self.linear1_bn = nn.BatchNorm1d(S)
            # Layer 2
            self.linear2 = nn.Linear(S, S)
            self.linear2_bn = nn.BatchNorm1d(S)
            # Layer 3
            self.linear3 = nn.Linear(S, S)        
            self.linear3_bn = nn.BatchNorm1d(S)
            # Layer 4
            self.linear4 = nn.Linear(S, S)        
            self.linear4_bn = nn.BatchNorm1d(S)
            # Layer 5
            self.linear5 = nn.Linear(S, S)        
            self.linear5_bn = nn.BatchNorm1d(S)
            # Output Layer
            self.linearO = nn.Linear(S, a_size)
            # Activation
            self.act = torch.relu
            # Dropout
            self.drop = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.linear1_bn(self.act(self.linear1(x)))
            x = self.drop(self.linear2_bn(self.act(self.linear2(x))))
            x = self.drop(self.linear3_bn(self.act(self.linear3(x))))
            x = self.drop(self.linear4_bn(self.act(self.linear4(x))))        
            x = self.drop(self.linear5_bn(self.act(self.linear5(x))))
            x = self.drop(self.linearO(x)) 
            return torch.sigmoid(x)

    model = MLP().to(device)
    model.load_state_dict(torch.load(saveddata))
    model.eval()
    
    # Initiate y_pred
    y_pred = torch.zeros(1, 1)
    y_logits = torch.zeros(1, 1)
    with torch.no_grad():
        for batch, inputs in enumerate(x):
        
            # Resize inputs
            x_tensor = inputs.view(inputs.shape[0], resize*resize)
            
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

y_pred, y_logits = predict(x_val)
y_pred = np.array(y_pred)
y_logits = np.array(y_logits)





# Set up target variable for valing
labels = []
for file in y_val:
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






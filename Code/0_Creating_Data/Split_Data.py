# Import packages
import os
import glob
import numpy as np  
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array, save_img

# Collect file names
GoT = '1_GoT/'
# Name of sub-folder for later
Other = glob.glob('0_*/')

x1 = []
for file in os.listdir(GoT):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"): 
        x1.append(GoT + filename)
x0 = []
for folder in Other:
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"): 
            x0.append(folder + filename)
y1 = []
for file in os.listdir(GoT):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"): 
        y1.append(GoT + filename)
y0 = []
for folder in Other:
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            y0.append(folder + filename)
            
x = x1 + x0
y = y1 + y0

# Set sklearn seed
seed = 2020

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    random_state=seed, test_size=0.3)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                    random_state=seed, test_size=0.5)
                                                    
val = "val/"

i = 1
for lab, im in zip(y_val, x_val):
    # Labels
    label = open(lab, "r")
    label = list(label.read().split('\n'))
    if label[0] == "Game of Thrones":
        label[0] = "1 Game of Thrones"
        filename = "1_Game_of_Thrones_"
    else:
        label[0] = "0 Sitcom"
        filename = "0_Sitcom_"
    trainlab = open(os.path.join(val, filename + str(i) + ".txt"), mode="w")
    trainlab.write(label[0])
    trainlab.close()
    # Images
    image = load_img(im)
    image = img_to_array(image)
    save_img(os.path.join(val, filename + str(i) + ".jpg"), image)
    i += 1       

test = "test/"

i = 1
for lab, im in zip(y_test, x_test):
    # Labels
    label = open(lab, "r")
    label = list(label.read().split('\n'))
    if label[0] == "Game of Thrones":
        label[0] = "1 Game of Thrones"
        filename = "1_Game_of_Thrones_"
    else:
        label[0] = "0 Sitcom"
        filename = "0_Sitcom_"
    trainlab = open(os.path.join(test, filename + str(i) + ".txt"), mode="w")
    trainlab.write(label[0])
    trainlab.close()
    # Images
    image = load_img(im)
    image = img_to_array(image)
    save_img(os.path.join(test, filename + str(i) + ".jpg"), image)
    i += 1     
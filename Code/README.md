# Final-Project-Group5
Final Project for Machine Learning II:  Deep Learning

## Get Data from Google Storage
Run these linux commands in your cloud terminal

wget https://storage.googleapis.com/game_of_thrones_classification/train.zip
unzip train.zip
wget https://storage.googleapis.com/game_of_thrones_classification/test.zip
unzip test.zip
wget https://storage.googleapis.com/game_of_thrones_classification/val.zip
unzip val.zip

## Create Dataset
If you prefer to recreate the data from scratch instead of fetching it from Google storage, create the necessary folders and run the code in [0_Creating_Data](0_Creating_dData/)

The R Markdown code [Webscrapping.Rmd](0_Creating_dData/Webscrapping.Rmd) will scrap the images from IMDB (again, noted it does not create folders so you will need to do this before hand).

The python code [Split_Data.py](0_Creating_dData/Split_Data.py) will relabel it as "1 Game of Thrones" or "0 Sitcom" and output it into the folders train, test, and val.

## Run Models

There are three subfolders in the Code folder [1_Epochs_25](1_Epochs_25/), [2_Epochs_300](1_Epochs_25/), [3_Epochs_500](1_Epochs_25/).  Each folder contains the syntax for each run.  

The folders are named based on the amount of epochs ran.  One can start at 25 and proceed to 300 and then 500.

Each file is named after its model type and main parameters.  For example, [cnn_lr1e-2_ep25_siebelm.py](1_Epochs_25/cnn_lr1e-2_ep25_siebelm.py) is a CNN model with a learning rate of 1e-2 and 25 epochs.

Each script loads the data from the train, test, and val folders.  It then models the data, saves the model as a ".pt" file, and runs a prediction funtion on either the test data or validation data.

Folder 3_Epochs_500 contains the main model and three alternative models.

[cnn_lr1e-8_ep500_siebelm.py](3_Epochs_500/cnn_lr1e-8_ep500_siebelm.py) contains the main model
[cnn_lr1e-8_ep500_NoOvsp_siebelm.py](3_Epochs_500/cnn_lr1e-8_ep500_NoOvsp_siebelm.py]) contains the non-oversampled alternative model
[cnn_lr1e-8_ep500_3layers_siebelm.py](3_Epochs_500/cnn_lr1e-8_ep500_3layers_siebelm.py) contains the three layer alternative model
[mlp_lr1e-8_ep500_siebelm.py](3_Epochs_500/mlp_lr1e-8_ep500_siebelm.py) contains the multi-layer perceptron alternative model
# Notebook README

## Overview
This folder contains all notebooks needed to run the model. 
The following is a detailed explanation of all the notebooks, their functionalities and how to use them. 
They are designed to be run in series, however if the necessary preprocessing steps have already been performed then you can only run 04_run_training.

## config_insecteffnet.yaml:

This is a YAML file containing the main configuration settings for the neural network, the spectrogram calculation, the training, data augmentation, and so on. It is loaded by some of the notebooks. You can experiment with many of the parameters in here if you want to fine-tune performance.

## 01_preprocess_waves.ipynb:
Preprocesses waveform data to uniform length and saves the data locally. 
The notebook is a refactored and adapted version of Marius Faiß pre-processing [script](https://github.com/mariusfaiss/InsectSet47-InsectSet66-Adaptive-Representations-of-Sound-for-Automatic-Insect-Recognition/blob/main/SplitAudioChunks.py).
Possible scenarios are that a sound file is either shorter or longer than a pre-defined uniform length, e.g. 5 seconds.
In case the audio is shorter, two versions are created that are padded with zeros or looped to the full defined length, respectively.
In case the audio is longer, it is chunked to pre-defined audio lengths that can overlap.
Hence, to run the notebook, a set of window lengths and overlaps have to be defined.
A reasonable heuristic is to choose the overlap as half of the window length.
We experimented with values ranging from 3.5-7.5 seconds.
Our preferred model was trained on 5 second windows with 2.5 overlaps.
A 3.5 second window with 1 second overlap was also amongst the top submissions.

We have now set this notebook to generate only 1 set of windows and overlaps (the `windows` variable in the code), but you could try others.

## 02_classweights.ipynb:
Calculates different types of class weights. 
Uses metadata from the "data"-folder as an input and saves class weights in an *.npy file in the "class_weights"-folder.
We experimented with different approaches to calculating weights, in particular:

     1: result = total_number_of_files / number_of_files_per_class
     2: result = 1 - (number_of_files_per_class / total_number_of_files)
     3: result = total_number_of_files / ((number_of_classes) * number_of_files_per_class)

Recommendations: Both approach 1. and 2. yield excellent performance. The most performant model relied on version 2.


## 03_scan_lr.ipynb:
The learning rate is one of the most important hyperparameters to tune during when training a deep learning model.
This notebooks provides a quick interface to the learning rate tuner implemented in Pytorch Lightning.
A grid of learning rates can be defined and the tuner calculates the loss of a few batches to estimate a good initial starting rate.
The results can be visualized and a final learning rate be suggested.
During regular training runs, e.g. in 04_run_training.ipynb, a learning rate scheduler (CosineAnnealing) is being used to smoothly
adjust the learning rate over the entire training run.


## 04_run_training.ipynb:
The main notebook to perform fine-tuning of a pre-trained model or to train a model from scratch.
All parameters of the training run can be set at the top of the notebook and are automatically saved as a .yaml file at the 
beginning of the training.
The training performance is logged using tensorboard and the best (in terms of validation F1), as well last model 
checkpoint are being saved.
After the model training, validation and test files are being predicted and results averaged over all audio chunks available 
for each file, respectively.
Error analysis is automatically being performed by calculating error metrics of the validation set and plotting of a
confusion matrix.


## 05_scan_parameters.ipynb:
This notebook acts as a convenience to loop over multiple hyperparameters to estimate the best settings.

# Insect sound classifier: InsectEffNet
Deep learning classification for insect sounds (grasshoppers, crickets, cicadas) developed by the winning team of the Capgemini Global Data Science Challenge 2023.

The classifier was trained to identify 66 species. The winning team (team name "It is a bug", Capgemini I&D Austria) developed a solution and training pipeline based on EfficientNet applied to audio spectrograms. This repository contains code to create, train and evaluate such ML models.

This README file provides an overview of the model, its functionality and how to use it. Refer to README files located in the sub-directories for in-depth explanations.

## Table of contents
1. [Setup](#setup)
   1. [Repository Structure](#repository-structure)
   2. [Usage](#usage)
2. [Methodology](#methodology)
   1. [Data](#data)
   2. [Model Description](#model-description)
   3. [Class Imbalance and Augmentations](#class-imbalance-and-augmentations)
   4. [Pre-trained vs. From Scratch](#pretrained-vs-trained-model)
   5. [Model Evaluation](#model-evaluation)
   6. [Hardware & Performance](#hardware--performance)
3. [Contributors & Acknowledgements](#contributors--acknowledgements)
4. [Licenses](#licenses)


## Setup

### Repository Structure
This is the expected structure to run the model successfully. 
Paths with no file ending are folders.

~~~
class_weights                       Directory containing different class weights, based on various metrics.
data                                Directory containing train, validation and test data.
notebooks/
  01_preprocess_waves.ipynb         Notebook for preprocessing sound data to uniform length. 
  02_classweights.ipynb             Notebook that calculates various class weights. 
  03_scan_lr.ipynb                  Notebook that trains the model multiple times with different learning rates.
  04_run_training.ipynb             Notebook that trains and evaluates the data. 
  05_scan_parameters.ipynb          Notebook that loops over multiple parameters for training
shell_scripts                       Directory that contains 2 shell scripts, to help reduce costs on AWS.
src/
  custom/
    __init__.py
    data.py
    eval.py
    net.py
    trainer.py
    utils.py
  config.py                         Imported libraries and utilities
  gdsc_utils.py                     Imported libraries and utilities
requirements.txt
~~~

### Usage
1. Clone this repository.
2. Create a Python virtual environment in which to do this work, e.g. using Conda or Virtualenv. (The code has been tested with Python 3.8 on Ubuntu, using virtualenv to create the virtual environment: `python -m venv venv_itisabug`)
3. After acitivating your virtual environment, install the necessary Python packages specified in `requirements.txt`, e.g. using ``pip install -r requirements.txt``
4. Make sure that the data folder is set up correctly (download the needed data, and create folders as needed). See the [README](./data/README.md).file inside the `data` subfolder for full instructions.
5. Navigate to the notebooks directory and read the [README](./notebooks/README.md).
6. Decide which hyperparameters to use, whether to use a pretrained model, etc.
7. Run the necessary notebooks including `04_run_training`.

Please be warned that some config settings and file paths might need to be adapted, depending on your OS and environment setup. 


## Methodology

### Data
The model is expecting waveform files in the data folder together with a metadata csv file, that contains the exact filename, path and label for training/validation data and filename, path for test data. 
Crucially, all data is supposed to have the same sampling frequency, but can vary in length.
We split every audio file into smaller windows of uniform length. 
We made this decision since we noticed a big deviation of audio length between our training data files and the spectrogram approach relies on audio files of the same length.

The notebook `01_preprocess_waves` contains the steps needed to prepare a dataset for use.

### Model Description
The ML model is built using PyTorch [Lightning](https://www.pytorchlightning.ai/index.html) and utilizes pre-trained models as part of its evaluation. 
It is based on the spectrogram approach of audio classification i.e. transforming raw waveform data into Mel spectrograms and then solving the resulting image classification problem. 
We use the small, on ImageNet21k pre-trained, [EfficientNetv2](https://github.com/google/automl/tree/master/efficientnetv2) architecture to solve the computer vision problem, which is as the name implies quite efficient to fine-tune and run. 
EfficientNetV2-S consists of ~20 million parameters, thus achieves fast training speed while maintaining state of the art performance. 
Link to the supporting research: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

### Class Imbalance and Augmentations

To counter class imbalance, custom class weights can be passed to the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
in Pytorch, allowing to weigh underrepresented classes higher or overrepresented ones lower, depending on how the weights are calculated.
For further details, we refer to the [02_classweights](./notebooks/02_classweights.ipynb) notebook.
To further reduce overconfident model predictions during training, we use label smoothing of 0.1 in the [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss),
however, the value can be tuned as necessary.

As data augmentations, we differentiate between augmentations that are applied on waveforms and on spectrograms. 
All augmentations are applied on the fly during training with a pre-defined probability per sample or batch.
The most performant results have been achieved with moderate levels of augmentations, with 10-15% chance of being applied
to solely waveforms. However, the options to additionally include the spectrogram augmentations listed below are 
still available in all training notebooks.

Waveform:
- OneOf(Noise Injection, Gaussian Noise, Pink Noise)
- Impulse response, i.e. to model reverb and room effects

Spectrogram:
- Mixup
- OneOf(MaskFrequency, MaskTime)


### Pretrained vs Trained Model
Decide if you want to fine-tune a pretrained model, or train a model from scratch. 
This and other hyperparameters can be set in the 04_run_training notebook.

Running the 04_run_training notebook creates sub-folders for each training run, containing model checkpoints, hyperparameter savefiles, events.out.tfevents for logging, two prediction csv and other useful files. Refer to the notebooks [README](./notebooks/README.md).

### Model Evaluation
We evaluate a single audio file by calculating the prediction of each uniform subfile, then averaging all predictions for that singular file. Finally, we calculate the f1-score, among other metrics, based on all predictions. In addition we return the confusion matrix and use Tensorboard logging, which is a supported logger for Pytorch Lightning, during our training runs. The saved events.out.tfevents files can be analyzed in tensorboard, to monitor all desirable metrics during the training run. For further details, refer to [tensorboard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.) documentation.

### Hardware & Performance

The following performance stats correpsond to the use of an `ml.g4dn.xlarge` instance on AWS Sagemaker, which uses a Tesla-V4 GPU.
The model only requires ~1.2 GB of GPU memory itself.
See the table below for details on memory usage and inference time.
Values have been calculated using 5 second long mel spectrograms (n_bins=128).
Results may vary for longer audio files or larger spectrograms.

| Batch Size              | GPU Memory [MB]         | Inference Time [ms] |
|-------------------------|---------------------|---------------------| 
| 1     | 1200      | 25.1                | 
| 32     | 4060      | 65.3                | 
| 64 | 6950 | 121                 |
| 128 | 12180 | 238                 |




## Contributors & Acknowledgements
List of people, that contributed in creating this model:
- Raffaela Heily
- Lukas Kemetinger
- Dominik Lemm
- Lucas Unterberger

Special thanks to the team of **Naturalis Biodiversity Center** that provided the data, the idea and the path towards a more sustainable and biodiverse future:
- Dr. Elaine van Ommen Kloeke 
- Max Schöttler
- Dr. Dan Stowell
- Marius Faiß

Many thanks to our Capgemini Sponsors and organizational team that made all of this possible:
- Niraj Parihar
- Susanna Ostberg
- TP Deo
- Anne-Laure Thiuellent
- Andris Roling
- Dr. Daniel Kühlwein
- Kanwalmeet Singh Kochar
- Kristin O'Herlihy
- Marc Niedermeier
- Mateusz Gryz
- Nikolai Babic
- Sebastian Sell
- Sophie (Nien-chun) Yin
- Steffen Klempau
- Surobhi Deb
- Timo Abele
- Tomasz Czerniawski

Additionally, we want to thank **AWS**, for providing us with their state of the art cloud infrastructure and enough resources, to make this challenge a reality.

## Licenses
The used impulse response data is available on [OpenAir](https://www.openair.hosted.york.ac.uk/) licenced under CC BY 4.0.
All additional Python dependencies are licensed either under MIT or Apache 2.0.

For the software license that covers this repository and the InsectEffNet code, see the file LICENSE.


# Data README 

This folder contains the whole data that is used for the training, evaluation and testing of the selected ML model as well as a metadata csv file. For detailed information look at the **Data** section below.

To run our standard experiment, you will need to download from two data sources:

1. "InsctSet66" from https://zenodo.org/records/8252141
2. Impulse responses (used during data augmentation) from https://www.openair.hosted.york.ac.uk/

These are described below.

## File and folder layout
This is the expected structure for accessing the given data within the notebooks. Paths with no file ending are folders.

~~~
data/
  irs                     Directory containing impulse response audio files.
  production_data/        Directory containing customized data.
    crop-x-s/             Directory with data at a given uniform length of x seconds
       train              Directory containing the customized validation data.
       val                Directory containing the customized validation data.
       metadata.csv       Metadata of the customized data.
    ...
  metadata.csv            Metadata of the training and validation data. -- This file will be generated.
  classlist.csv           List of the classes and the species names they correspond to. -- This file will be generated.
~~~

## Insect audio recordings: InsectSet66

"InsctSet66" is available at https://zenodo.org/records/8252141 . Download the zip of wavs, and the csv of metadata (these are separate downloads). Unzip the WAVS. (You will need 13 GB of disk space for the zip plus the unzipped audio.)

You can store this anywhere you like. In the notebook file we assume that it has been unzipped to `data/InsectSet66` inside the repository, but it can be elsewhere -- just change the path used in that notebook.

## Impulse Responses (IR)

To use IR augmentation with the model, additional files have to be downloaded. 
We used IR's from [OpenAir](https://www.openair.hosted.york.ac.uk/) under the IR tab.
Select an environment and download the audio files under the "Impulse Responses" tab.

IR's used in this work are:
- [Gill Heads Mine](https://www.openair.hosted.york.ac.uk/?page_id=494)
- [Koli National Park Summer](https://www.openair.hosted.york.ac.uk/?page_id=577)
- [Koli National Park Winter](https://www.openair.hosted.york.ac.uk/?page_id=584)
- [Troller's Gill](https://www.openair.hosted.york.ac.uk/?page_id=745)
- [Tyndall Bruce Monument](https://www.openair.hosted.york.ac.uk/?page_id=764)

Inside the `data` folder, create a subfolder `irs`, and inside that a subfolder `openair`. Move all download IR files there.
An example path should have the following depth: ``irs/*/*/mono/*.wav``.

## Data file layout and preprocessing
Once the audio and the IRs are downloaded, you should be ready to run the preprocessing notebook script which generates the audio files that will actually be used to train the classifier.

NB we expect all the audio files to have the same sampling frequency, but can vary in length. If you use a different dataset you may need to check this.

After using the 01_preprocess_waves notebooks a folder with data that contains waveforms at a given uniform length is created and placed into the production_data folder. This folder contains a metadata for the created files.
For more detailed information look at the [README](https://github.com/Dom1L/GDSC23/blob/main/notebooks/README.md) in the notebooks folder.
The metadata for this customized files has a structure given as follows:

| file_name | unique_file | path | label | subset |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| insect_dae_003.wav | insect_dae_003 | data/production_data/crop-x-s/val/insect_dae_003_chunk1.wav | 49 | validation | 
| insect_dae_003.wav | insect_dae_003 | data/production_data/crop-x-s/val/insect_dae_003_chunk1.wav | 49 | validation | 
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/production_data/crop-x-s/val/insect_bcd_001_dat1_loop.wav | 34 | validation | 
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/production_data/crop-x-s/val/insect_bcd_001_dat1_padded.wav | 34 | validation | 
| insect_abc_001.wav | insect_abc_001 | data/production_data/crop-x-s/train/insect_abc_001_chunk1.wav | 1 | train |

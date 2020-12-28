# STS autoencoder

This model is currently under construction.

The model is constituted of 3 parts:
1. dB-Mel data encoder
	- encoder takes the spectrogram data to a GRU network then to a set of convolution layers
	- I expect this module could be used to other models where voice data is required to be introduced somehow
2. attention-based encoder-to-text module
	- where the module compares the results to one-hot encoded text data
3. Convolution-based decoder
	- expected to output a dB-Mel-like data

I am trying to keep the model as small as possible as I expect this model to be used along with other models, not by itself.

TODO
	- add preprocessing module for writing-systems other than Hangul
	- modify the `tester.py` as it is made for the previous model of this repository
	- modify the `plotter.py`

## Preprocessing unit
I tried to design the module so that the preprocessing unit is essentially separated from the main training model. The subfolder `preprocessing_unit` contains the following two files:
	- `text_preprocess.py`
		1. preprocesses the texts
		2. returns the dictionary of text to one-hot-vector as a pickle file (for Korean Hangul, the program outputs the dictionary to `encoded_hangul.pkl` file)
	- `wav_and_text_preprocess.py`
		1. converts audio and text files to a set of numpy arrays
		2. outputs the arrays as npz files
		3. also outputs the list of audio and text npz filename-pairs
			- filename: `filelist_pairs.csv`

## Explanation of the STS algorithm used
The `utils.py` contains the NN models, loss functions and other necessary modules where the `STS.py` model trains the NNs. \
There are 3 neural networks in this STS algorithm
- encoder module: is used for both STT and the main STS models
- totext module: takes an output of the encoder and outputs an one-hot-vector
- decoder module: takes an encoder output and outputs a dB-Mel Spectrogram data  

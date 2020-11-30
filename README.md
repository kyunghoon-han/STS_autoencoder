# STS_autoencoder

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



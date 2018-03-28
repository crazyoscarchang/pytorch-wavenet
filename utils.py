import numpy as np
from scipy.io.wavfile import read
import torch

def bin_wav(wavfile, nbins):
    frequency, data = read(wavfile)
    # np.min is much faster than built-in min/max
    bins = np.linspace(data.min(), data.max(), nbins)
    # searchsorted is also much faster than digitize for large number of bins
    # because it uses a binary search tree where digitize uses linear search
    return np.searchsorted(bins, data, "right")

def bin_and_one_hot(wavfile, nbins):
    return np.eye(nbins + 1)[bin_wav(wavfile, nbins)]

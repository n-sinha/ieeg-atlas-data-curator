"""
    This script gets coherence networks and bandpower for a clip of iEEG
"""
from fractions import Fraction
import itertools

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, get_window, welch, coherence, resample_poly
from scipy.integrate import simpson
from .format_network import format_network

bands = [
    [0.5, 4],  # delta
    [4, 8],  # theta
    [8, 12],  # alpha
    [12, 30],  # beta
    [30, 80],  # gamma
    [0.5, 80],  # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)

#%%
def get_features_from_data(existing_data):
    """This function generates preictal features from a clean dataset already loaded"""

    data = existing_data
    fs = int(np.around(1 / (data.index[1] - data.index[0])))
    (n_samples, n_channels) = data.shape

    # calculate psd
    window = get_window("hamming", fs * 2)
    freq, pxx = welch(x=data, fs=fs, window=window, noverlap=fs, axis=0)

    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))

    cohers = np.zeros((len(freq), n_edges))

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        _, pair_coher = coherence(
            data.iloc[:, ch1],
            data.iloc[:, ch2],
            fs=fs,
            window="hamming",
            nperseg=fs * 2,
            noverlap=fs,
        )

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    pxx = pxx[filter_idx]
    cohers = cohers[filter_idx]

    pxx_bands = np.empty((N_BANDS, n_channels))
    coher_bands = np.empty((N_BANDS, n_edges))

    pxx_bands[-1] = np.log10(simpson(pxx, dx=freq[1] - freq[0], axis=0) + 1)
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)

        pxx_bands[i_band] = simpson(pxx[filter_idx], dx=freq[1] - freq[0], axis=0)
        pxx_bands[i_band] = np.log10(pxx_bands[i_band] + 1)

        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    pxx_bands[:-1] = pxx_bands[:-1] / np.sum(pxx_bands[:-1], axis=0)

    network_bands = format_network(coher_bands, n_channels)

    return pxx_bands, network_bands

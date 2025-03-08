"""
    This script gets coherence networks and bandpower for a clip of iEEG
"""
from fractions import Fraction
import itertools

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, get_window, welch, coherence, resample_poly
from scipy.integrate import simpson

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
def get_network_coherence_from_data(existing_data):
    """This function generates preictal features from a clean dataset already loaded"""

    data = existing_data
    cols = data.columns

    fs = int(np.around(1 / (data.index[1] - data.index[0])))
    (n_samples, n_channels) = data.shape

    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))

    cohers = []

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        freq, pair_coher = coherence(
            data.iloc[:, ch1],
            data.iloc[:, ch2],
            fs=fs,
            window="hamming",
            nperseg=fs * 2,
            noverlap=fs,
        )
        cohers.append(pair_coher)

    cohers = np.array(cohers).transpose()
    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    cohers = cohers[filter_idx]

    coher_bands = np.empty((N_BANDS, n_edges))

    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)
        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    ix = pd.MultiIndex.from_tuples(
        [(cols[i], cols[j]) for (i, j) in itertools.combinations(range(n_channels), 2)],
        names=["channel_1", "channel_2"],
    )
    coher_bands = pd.DataFrame(coher_bands.transpose(), index=ix, columns=band_names)

    return coher_bands

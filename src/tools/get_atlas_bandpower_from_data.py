"""
    This script gets coherence networks and bandpower for a clip of iEEG
"""
from fractions import Fraction
import itertools

import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, get_window, welch, coherence, resample_poly

# from scipy.integrate import simpson

# from .get_iEEG_data import get_iEEG_data
# from .common_avg_reference import common_avg_reference
# from .butter_bp_filter import butter_bp_filter
# from .format_network import format_network

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
def get_atlas_bandpower_from_data(existing_data):
    """This function generates preictal bandpower from a clean dataset already loaded"""

    data = existing_data
    fs = int(np.around(1 / (data.index[1] - data.index[0])))
    (n_samples, n_channels) = data.shape

    # calculate psd
    window = get_window("hamming", fs * 2)
    freq, pxx = welch(x=data, fs=fs, window=window, noverlap=fs, axis=0)

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    pxx = pxx[filter_idx]
    # pxx = np.divide(pxx, np.sum(pxx, 0))
    pxx = pd.DataFrame(pxx, index=freq, columns=data.columns)

    return pxx

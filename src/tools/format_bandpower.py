"""
    This script integrates bandpower into canonical bands
"""


import numpy as np
import pandas as pd
from scipy.integrate import simpson

#%%
def format_bandpower(pxx):

    bands = [
        [0.5, 4],  # delta
        [4, 8],  # theta
        [8, 12],  # alpha
        [12, 30],  # beta
        [30, 80],  # gamma
        [0.5, 80],  # broad
    ]
    band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]

    N_BANDS = len(band_names)
    freq = pxx['freq'].to_numpy()
    pxx = pxx['bandpower'].to_numpy()
    pxx_bands = np.empty(N_BANDS)

    pxx_bands[-1] = np.log10(simpson(pxx, dx=freq[1] - freq[0], axis=0) + 1)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)

        pxx_bands[i_band] = simpson(pxx[filter_idx], dx=freq[1] - freq[0], axis=0)
        pxx_bands[i_band] = np.log10(pxx_bands[i_band] + 1)

    pxx_bands[:-1] = pxx_bands[:-1] / np.sum(pxx_bands[:-1], axis=0)

    pxx_bands = pd.Series(
        pxx_bands, index=pd.Index(band_names, name="band"), name="power"
    )
    return pxx_bands

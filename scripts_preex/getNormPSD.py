import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming

def get_norm_psd(iEEGnormal, data_timeS, SamplingFrequency):
    # Set sampling frequency, time domain data, window length, and NFFT
    Fs = SamplingFrequency
    data_seg = data_timeS[:Fs*60, :]
    window = Fs * 2
    NFFT = window

    # Compute PSD
    f, psd = welch(data_seg, fs=Fs, window=hamming(window), nperseg=window, noverlap=0, nfft=NFFT, axis=0)

    # Filter out noise frequency between 57.7Hz and 62.5Hz
    psd = np.delete(psd, np.where((f >= 57.7) & (f <= 62.5)), axis=0)
    f = np.delete(f, np.where((f >= 57.7) & (f <= 62.5)))

    # Compute bandpower
    def bandpower(psd, freqs, freq_range):
        idx_band = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
        return np.trapz(psd[idx_band], freqs[idx_band])

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80),
        'broad': (1, 80)
    }

    band_powers = {band: bandpower(psd, f, freq_range) for band, freq_range in bands.items()}
    band_powers_log = {band + 'log': np.log10(value + 1) for band, value in band_powers.items()}

    # Calculate total power and relative band powers
    total_power = sum(band_powers_log.values())
    relative_band_powers = {band + 'Rel': value / total_power for band, value in band_powers_log.items()}

    # Append results to iEEGnormal DataFrame
    data_to_append = {**band_powers_log, **relative_band_powers}
    iEEGnormal = pd.concat([iEEGnormal, pd.DataFrame(data_to_append, index=iEEGnormal.index)], axis=1)

    return iEEGnormal

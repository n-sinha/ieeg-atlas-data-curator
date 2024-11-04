import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, welch, iirnotch

def eeg_filter(data, cutoff, fs, btype, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def notch_filter(data, freq, fs):
    notch_freq = freq / (0.5 * fs)
    b, a = iirnotch(notch_freq, Q=30)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def get_norm_entropy(iEEGnormal, data_timeS, SamplingFrequency):
    Fs = SamplingFrequency
    data_seg = data_timeS[:Fs*60, :]

    # Filter the data
    data_seg_low = eeg_filter(data_seg, 80, Fs, 'low')
    data_seg_high = eeg_filter(data_seg_low, 1, Fs, 'high')
    data_seg_notch = notch_filter(data_seg_high, 60, Fs)

    # Compute Shannon entropy
    entropy = np.array([-np.sum(signal * np.log2(signal + 1e-10)) for signal in data_seg_notch.T])
    entropy_log = np.log10(-entropy + 1)

    # Append entropy data to DataFrame
    iEEGnormal['entropy'] = entropy_log
    return iEEGnormal

# %%
import numpy as np
import os, sys
import mne
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

#%%
def plot_eeg(data_df, fs):
    """
    Plot EEG data using MNE.
    
    Args:
        data_df (pd.DataFrame): DataFrame with EEG data (channels as columns, time as rows)
        fs (float): Sampling frequency in Hz
    """
    # Create MNE info object
    labels = list(data_df.columns)
    info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=['eeg'] * len(labels))
    
    # Create MNE raw object
    raw = mne.io.RawArray(data_df.values.T, info)
    
    # Plot with interactive settings
    raw.plot(
        scalings='auto',
        n_channels=len(labels),
        title='EEG Recording',
        show=True,
        block=True,
        duration=10,
        start=0
    )

# Example usage
if __name__ == "__main__":
    # Initialize path to data
    PKL_PATH = '/Users/nishant/Dropbox/Sinha/Lab/Research/ieeg_atlas_harmonization/data/hup/derivatives/bipolar/sub-RID0031/interictal_eeg_bipolar_10.pkl'
    eeg = np.load(PKL_PATH, allow_pickle=True)
    plot_eeg(eeg, fs=200)



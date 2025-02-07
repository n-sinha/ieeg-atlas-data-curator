# %%
import numpy as np
import os, sys
import mne
# need to install pyqt5/6, pip install pyqt6

# the trick to have interactive plots
%matplotlib qt 

# %%
eeg = np.load('/Users/nishant/Dropbox/Sinha/Lab/Research/ieeg_atlas_harmonization/data/hup/derivatives/bipolar/sub-RID0031/interictal_eeg_bipolar_0.pkl', allow_pickle=True)


# %%

# load data in or read edf files
key = '15'
e = int(key)-1
plot_pred = True
module = 'sparcnet-update'
add_1st = False
data = eeg.values
labels = list(eeg.columns)  # Convert pandas Index to list
fs = 200
valid_index = np.all(~np.isnan(data),axis=1)
data[~valid_index,:] = 0

# use mne to plot the data
info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=['eeg'] * len(labels))
raw = mne.io.RawArray(data.T, info)  # Note: data needs to be transposed

# Plot with interactive settings
raw.plot(
    scalings='auto',  # Automatically scale the data
    n_channels=len(labels),  # Show all channels
    title='EEG Recording',
    show=True,
    block=True,  # This ensures the plot window stays open
    duration=20,  # Show 20 seconds at a time
    start=0  # Start from the beginning
)

# Keyboard shortcuts to remember:
# + or - : Increase/decrease amplitude
# → or ← : Scroll through time
# Page Up/Down: Scroll through channels
# Home/End: Go to start/end of data
# z: Toggle zoom mode
# a: Toggle annotation mode

# %%

# create mne object, filter and set montage
info = mne.create_info(ch_names = old_ch_labels, sfreq = fs, ch_types = 'eeg')
raw = mne.io.RawArray(data.T*1e-6, info)
# raw.resample(256)
raw.filter(1., 40., method='iir',iir_params=dict(order=4, ftype='butter', output='sos'))
raw.notch_filter(freqs=[60],method='iir',iir_params=dict(order=4, ftype='butter', output='sos'))
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw = mne.set_bipolar_reference(raw,['Fp1','F7','T3','T5','Fp2','F8','T4','T6',
                    'Fp1','F3','C3','P3','Fp2','F4','C4','P4','Fz'],['F7','T3','T5','O1','F8','T4','T6','O2',
                    'F3','C3','P3','O1','F4','C4','P4','O1','Cz'])

# get start and duration of each event
sz_range = extract_seiz_ranges(labels)
sz_starts = np.array([a[0]/fs for a in sz_range])
sz_dura = np.array([(a[1]-a[0])/fs for a in sz_range])
sz_text = ['SS_Sz']*len(sz_starts)
# set annotations object
annotations = mne.Annotations(onset=np.concatenate([sz_starts]), duration=np.concatenate([sz_dura]), description=sz_text)


# Add annotations to raw object
raw.set_annotations(annotations)

raw.plot(title=key)



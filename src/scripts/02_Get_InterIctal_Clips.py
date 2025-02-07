# %% [markdown]
# # Get_Interictal_Clips
# - use erin conrad's code to get sleep times by alpha:delta ratio
# - use erin's sheet to get all seizure times
# - akash's code to find the best interictal times
# - this code downloads from the files in data/pt/interictal_times and files.npy

# %%
import os
import sys

code_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(code_path)

import json
from os.path import join as ospj
import warnings

import numpy as np
import pandas as pd
import tools
from tqdm import tqdm
from ieeg.auth import Session
from scipy.io import savemat, loadmat
from fractions import Fraction
from scipy.signal import (
    iirnotch,
    filtfilt,
    butter,
    get_window,
    welch,
    coherence,
    resample_poly,
)
from scipy.integrate import simpson

warnings.filterwarnings("ignore")

# %%
# Get paths from config file and metadata
with open(ospj("config.json"), "rb") as f:
    config = json.load(f)
repo_path = config["repositoryPath"]
metadata_path = ospj(repo_path, "ieeg-metadata")
data_path = ospj(repo_path, "data")

# from erin's sheet below
seizure_metadata = pd.read_excel(
    ospj(metadata_path, "Manual validation.xlsx"), sheet_name="AllSeizureTimes"
).dropna(how="all")

# credentials
USERNAME = config["usr"]
PWD_BIN_FILE = config["pwd"]

# %% [markdown]
# ## Run loop find best clips

# %%
patients = pd.read_csv(ospj(metadata_path, "patient_list.csv"))["patient"]


# %%
for pt in tqdm(patients, total=len(patients)):
    interictal_clips = np.load(ospj(data_path, pt, "interictal_clip_times.npy"))
    interictal_files = np.load(ospj(data_path, pt, "interictal_clip_files.npy"))

    for i_ii, (clip_start, fileID) in enumerate(
        zip(interictal_clips, interictal_files)
    ):
        clip_end = clip_start + 60

        try:
            data, fs = tools.get_iEEG_data(
                USERNAME,
                PWD_BIN_FILE,
                seizure_metadata.query("Patient == @pt & IEEGID == @fileID")[
                    "IEEGname"
                ].iloc[0],
                clip_start * 1e6,
                clip_end * 1e6,
            )
        except:
            try:
                data, fs = tools.get_iEEG_data(
                    USERNAME,
                    PWD_BIN_FILE,
                    seizure_metadata.query("Patient == @pt & IEEGID == @fileID")[
                        "IEEGname"
                    ].iloc[0],
                    clip_start * 1e6,
                    clip_end * 1e6,
                )
            except:
                print(f"skipped {pt} num{i_ii} due to IEEG.org error")
                continue

        clean_channels = tools.clean_labels(data.columns)
        # find and return a boolean mask for non ieeg channels
        non_ieeg_channels = tools.find_non_ieeg(clean_channels)

        data.columns = clean_channels
        data = data.iloc[:, ~non_ieeg_channels]

        data_ref = tools.automatic_bipolar_montage(data, data.columns)

        # bandpass between 0.5 and 80 and notch filter 60Hz
        data_bandpass = tools.butter_bp_filter(data_ref, 0.5, 80, fs)
        b, a = iirnotch(60.0, 30.0, fs)
        data_filtered = filtfilt(b, a, data_bandpass, axis=0)

        # downsample to 200 hz
        new_fs = 200
        frac = Fraction(new_fs, int(fs))
        data_resampled = resample_poly(
            data_filtered, up=frac.numerator, down=frac.denominator
        )
        fs = new_fs

        (n_samples, n_channels) = data_resampled.shape
        # set time array
        t_sec = np.linspace(clip_start, clip_end, n_samples, endpoint=False)

        data_resampled = pd.DataFrame(
            data_resampled, index=t_sec, columns=data_ref.columns
        )

        if not os.path.exists(ospj(data_path, pt, "interictal")):
            os.mkdir(ospj(data_path, pt, "interictal"))

        data_resampled.to_pickle(
            ospj(data_path, pt, "interictal", f"interictal_eeg_bipolar_{i_ii}.pkl")
        )

# %%

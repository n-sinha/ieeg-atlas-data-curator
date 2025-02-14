# %% Get_Interictal_Clips
# - use erin conrad's code to get sleep times by alpha:delta ratio
# - use erin's sheet to get all seizure times
# - find best times for inteictal clips

# %%
import os

import json
from os.path import join as ospj
import warnings

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

# %%
# Get paths from config file and metadata
with open("config.json", "rb") as f:
    config = json.load(f)
repo_path = config["repositoryPath"]
metadata_path = ospj(repo_path, "ieeg-metadata")
data_path = ospj(repo_path, "data")

# from erin's sheet below
seizure_metadata = pd.read_excel(
    ospj(metadata_path, "Manual validation.xlsx"), sheet_name="AllSeizureTimes"
).dropna(how="all")

# from FC toolbox intermediate output code
with open(ospj(metadata_path, "sleep_times.pkl"), "rb") as f:
    sleep_times = pickle.load(f)

all_patients = sleep_times["name"]
all_files = sleep_times["file"]
all_sleep = sleep_times["sleep"]
all_times = sleep_times["times"]

patient_list = pd.read_csv(ospj(metadata_path, "patient_list.csv"))['patient']

n_patients = all_patients.shape[0]

# constants
PREICTAL_WINDOW_USEC = 30 * 1e6
DATA_PULL_MIN = 5  # size of window for each downloaded data chunk
F0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor

IMPL_EFF_BUFF = 72 * 60 * 60
SZ_BUFF = 2 * 60 * 60
BEFORE_BUFF = 2 * 60 * 60


# %% [markdown]
# ## define functions
# - first block references
# - second block contiguous regions
# %% function to get longest segments of interictal
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    # idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] - 1  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


# %% [markdown]
# ## Run loop find best clips

# %%
# %%
# %%
for i_pt in range(n_patients):
    patient = all_patients[i_pt]
    file = np.squeeze(all_files[i_pt])
    time = np.squeeze(all_times[i_pt])

    # remove this line to run on all patients (not just the ones in the atlas cohort)
    if patient not in patient_list.tolist():
        continue

    if not os.path.exists(ospj(data_path, patient)):
        os.makedirs(ospj(data_path, patient))

    # times that are during wake
    is_wake = ~np.squeeze(all_sleep[i_pt]).astype(bool)

    # times that are outside of seizures
    is_outside_sz = np.ones(time.shape, dtype=bool)

    seizures = seizure_metadata[seizure_metadata["Patient"] == patient]

    if np.unique(file).size == 1:
        # times that are not in implant effect window
        is_not_impl = time > IMPL_EFF_BUFF

        for i_sz, row in seizures.iterrows():
            sz_start = row["start"]
            sz_end = row["end"]

            is_before = np.logical_and(sz_start - BEFORE_BUFF < time, time < sz_start)

            AFTER_BUFF = SZ_BUFF

            is_after = np.logical_and(sz_end < time, time < sz_end + AFTER_BUFF)

            is_outside_sz = np.logical_and(
                is_outside_sz, np.logical_and(~is_before, ~is_after)
            )
    else:
        # times that are not in implant effect window
        is_not_impl = np.logical_or(time > IMPL_EFF_BUFF, file != 1)

        for i_sz, row in seizures.iterrows():
            sz_start = row["start"]
            sz_end = row["end"]
            file_num = int(row["IEEGID"])

            is_correct_file = file == file_num

            is_before = np.logical_and(sz_start - BEFORE_BUFF < time, time < sz_start)

            AFTER_BUFF = SZ_BUFF

            is_after = np.logical_and(sz_end < time, time < sz_end + AFTER_BUFF)

            is_outside_sz = np.logical_and(
                is_outside_sz, np.logical_and(~is_before, ~is_after)
            )
        is_outside_sz = np.logical_and(is_outside_sz, is_correct_file)

    # put all the conditions together
    interictal_mask = np.logical_and(
        is_wake, np.logical_and(is_not_impl, is_outside_sz)
    )

    # get segments where all values are true and sort by longest
    true_segments = contiguous_regions(interictal_mask)
    best_interictal_times = time[true_segments]
    best_interictal_files = file[true_segments]

    # keep only segments that are within one file
    keep_segments = best_interictal_files[:, 0] - best_interictal_files[:, 1] == 0

    best_interictal_times = best_interictal_times[keep_segments]
    interictal_files = best_interictal_files[keep_segments][:, 0]

    longest_segments = np.argsort(
        -1 * (best_interictal_times[:, 1] - best_interictal_times[:, 0]), kind="stable"
    )
    best_interictal_times = best_interictal_times[longest_segments]
    interictal_files = interictal_files[longest_segments]

    # get timestamps to search for interictal clips and save
    np.save(
        ospj(data_path, patient, "best_interictal_times.npy"), best_interictal_times
    )
    np.save(
        ospj(data_path, patient, "best_interictal_times_filenum.npy"), interictal_files
    )

    ii_win_len = 30
    clips = set()
    file_nums = np.array([])

    n_periods = 2
    n_clips_per_period = 10
    if len(best_interictal_times) == 1:
        n_periods = 1
        n_clips_per_period = 20

    for ii_period in range(n_periods):
        start = best_interictal_times[ii_period, 0]
        end = best_interictal_times[ii_period, 1]

        for _ in range(n_clips_per_period):
            rng = np.random.RandomState(2021)
            temp = rng.uniform(start, end)
            while any(
                temp >= existing_st and temp <= existing_st + ii_win_len
                for existing_st in clips
            ):
                temp = rng.uniform(start, end)
            clips.add(temp)
            file_nums = np.append(file_nums, interictal_files[ii_period])

    clips = np.array(list(clips))
    np.save(ospj(data_path, patient, "interictal_clip_times.npy"), clips)
    np.save(ospj(data_path, patient, "interictal_clip_files.npy"), file_nums)


# %%

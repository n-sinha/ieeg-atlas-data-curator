#%%
import pandas as pd
import typer
import mne
import matplotlib
matplotlib.use('qtagg')  # Set Qt backend (auto-detects Qt5/Qt6) before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import logging

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_not_empty(value):
    """Check if a value is not empty/missing"""
    if pd.isna(value):  # Handles NaN, None
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ['n/a', 'na', '', 'nan', 'null']
    return True

#%%

def main():
    # get all events.tsv files in the data/output/ram directory
    events_tsv_files = list(Path("data/output/ram").rglob("**/*ieeg*/*_events.tsv"))
    logging.info(f"Found {len(events_tsv_files)} events.tsv files")
    start_task_onsets = []
    for events_tsv_file in events_tsv_files:
        # read the events.tsv file
        events = pd.read_csv(events_tsv_file, sep="\t")
        # find the row where the value of stim_file is not empty
        non_empty_stim = events[events['stim_file'].apply(is_not_empty)]
        # find the location of the first non-empty stim_file and use that to get the onset
        start_task_onset = non_empty_stim['onset'].iloc[0]
        start_task_onsets.append(start_task_onset)
        logging.info(f"Onset: {start_task_onset} for {events_tsv_file.name}")

    logging.info(f"Minimum start task onset: {min(start_task_onsets)}")

#%%

if __name__ == "__main__":
    typer.run(main)

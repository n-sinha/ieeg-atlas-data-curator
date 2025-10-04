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

#%%

def main():
    # get all events.tsv files in the data/output/ram directory
    events_tsv_files = list(Path("data/output/ram").rglob("**/*ieeg*/*_events.tsv"))
    logging.info(f"Found {len(events_tsv_files)} events.tsv files")
    for events_tsv_file in events_tsv_files:
        # read the events.tsv file
        events = pd.read_csv(events_tsv_file, sep="\t")
        stim = events['stim_file'].unique()
        logging.info(f"Stim: {stim} for {events_tsv_file.name}")

#%%

if __name__ == "__main__":
    typer.run(main)

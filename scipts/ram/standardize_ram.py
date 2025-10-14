#%%
import pandas as pd
import typer
import mne
import matplotlib
matplotlib.use('qtagg')  # Set Qt backend (auto-detects Qt5/Qt6) before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import logging
import shutil
import nibabel as nib
from nibabel.affines import apply_affine

from pathlib import Path
from curate_pull_ram import clean_ram_metadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%
class StandardizeRAM:
    def __init__(self, openneuro_subject_dir: Path | str, channel_metadata: Path | str):            
        if Path(openneuro_subject_dir).is_dir():
            self.subject_dir = Path(openneuro_subject_dir)
        else:
            raise FileNotFoundError(f"Subject directory not found: {openneuro_subject_dir}")
        
        if Path(channel_metadata).is_file():
            self.channel_metadata = clean_ram_metadata(pd.read_csv(channel_metadata))
        else:
            raise FileNotFoundError(f"Channel metadata file not found: {channel_metadata}")

    def get_task_onsets(self) -> list:
        events_tsv_files = list(self.subject_dir.rglob("**/*ieeg*/*ses-0*_events.tsv"))
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
        return start_task_onset
    
    def standardize_directory_structure(self):       
        # Get subject ID
        subject_id = self.subject_dir.name

        # Create all primary directories
        (self.subject_dir.parent / "BIDS" / subject_id / 'primary' / subject_id / 'ses-postimplant').mkdir(parents=True, exist_ok=True)
        (self.subject_dir.parent / "BIDS" / subject_id / 'primary' / subject_id / 'ses-preimplant').mkdir(parents=True, exist_ok=True)
        (self.subject_dir.parent / "BIDS" / subject_id / 'primary' / subject_id / 'ses-postsurgery').mkdir(parents=True, exist_ok=True)

        # Create all derivative directories
        (self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg-clips' / 'ses-interictal').mkdir(parents=True, exist_ok=True)
        (self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg-clips' / 'ses-task').mkdir(parents=True, exist_ok=True)
        (self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg-recon').mkdir(parents=True, exist_ok=True)
        logging.info(f"Sirectory structure standardized for {subject_id} in {self.subject_dir.parent / 'BIDS'}")

    def curate_interictal_ieeg(self, start_time=0.0, end_time=135.0):

        # run the standardize_directory_structure method
        self.standardize_directory_structure()

        subject_id = self.subject_dir.name
        output_dir = self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg-clips' / 'ses-interictal'

        # get the edf files from ses 0 in the data/output/ram directory
        edf_files = list(self.subject_dir.rglob("**/*ieeg*/*ses-0*.edf"))
        logging.info(f"Found {len(edf_files)} edf files in bipolar montage from ses 0")

        for edf_file in edf_files:            
            # Build new filename: replace ses-0_task-FR1 with ses-interictal_run-XX
            filename = edf_file.name  # Get just the filename
            parts = filename.split('_')  # Split by underscore
            subject_part = parts[0]  # Extract subject ID (e.g., "sub-R1010J")
            
            # Find everything from "acq-" onwards and keep it
            acq_index = next(i for i, part in enumerate(parts) if part.startswith('acq-'))
            end_parts = '_'.join(parts[acq_index:])  # e.g., "acq-bipolar_ieeg.edf"
            
            # Build new filename with ses-interictal
            new_filename = f"{subject_part}_ses-interictal_{end_parts}"
            
            # Save the clipped edf file
            output_file = output_dir / new_filename
            
            # Skip if file already exists
            if output_file.exists():
                logging.info(f"File already exists, skipping: {output_file}")
                continue
            
            # read the edf file
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            # clip the edf file
            raw = raw.copy().crop(tmin=start_time, tmax=end_time)
            raw.export(output_file, fmt='edf')
            logging.info(f"Clipped edf file saved to {output_file}")

    def curate_task_ieeg(self):
        # move subject directory to the task directory
        subject_dir = self.subject_dir
        shutil.copytree(subject_dir, self.subject_dir.parent / "BIDS" / subject_dir.name / "derivatives" / "ieeg-clips" / "ses-task", dirs_exist_ok=True)
        logging.info(f"Subject directory moved to {self.subject_dir.parent / 'BIDS' / subject_dir.name / 'derivatives' / 'ieeg-clips' / 'ses-task'}")

    def curate_ieeg_recon(self):

        electrodes_tsv_files = list(self.subject_dir.rglob("**/*ieeg*/*ses-0*_electrodes.tsv"))

        electrodes = pd.DataFrame()
        
        for electrodes_tsv_file in electrodes_tsv_files:
            electrodes_temp = pd.read_csv(electrodes_tsv_file, sep="\t")
            electrodes_temp = electrodes_temp.filter(items=['name', 'tal.x', 'tal.y', 'tal.z'])
            electrodes = pd.concat([electrodes, electrodes_temp])

        df_surf = electrodes[['tal.x', 'tal.y', 'tal.z']]
        df_surf['colors'] = 1
        df_surf['size'] = 1
        df_surf['roi'] = electrodes['name']
        return df_surf, electrodes

#%%

def is_not_empty(value):
    """Check if a value is not empty/missing"""
    if pd.isna(value):  # Handles NaN, None
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ['n/a', 'na', '', 'nan', 'null']
    return True

def main():
    project_root = Path(__file__).parent.parent.parent
    channel_metadata = project_root / "data" / "input" / "ram" / "channel_metadata.csv"
    data_dir = project_root / "data" / "output" / "ram"
    standardize_ram = StandardizeRAM(openneuro_subject_dir=data_dir, channel_metadata=channel_metadata)
    # standardize_ram.curate_interictal_ieeg(start_time=0.0, end_time=135.0)
    # standardize_ram.curate_task_ieeg()
    df_surf, electrodes = standardize_ram.curate_ieeg_recon()
    df_surf.to_csv(project_root / "data" / "output" / "ram" / "electrodes2ROI.node", sep=' ', index=False, header=False)
    
#%%

if __name__ == "__main__":
    typer.run(main)

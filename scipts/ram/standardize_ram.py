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

    def curate_ieeg_recon(self, project_root: Path):
        self.standardize_directory_structure()
        subject_id = self.subject_dir.name
        output_dir = self.subject_dir.parent / "BIDS" / subject_id / "derivatives" / "ieeg-recon" / 'module4'
        output_dir.mkdir(parents=True, exist_ok=True)

        # MNI152 template path (using FreeSurfer templates)
        mni152_template = Path(project_root) / 'assets' / 'freesurfer' / 'cvs_avg35_inMNI152' / 'mri' / 'T1.mgz'
        mni152_template_nii = Path(project_root) / 'assets' / 'freesurfer' / 'cvs_avg35_inMNI152' / 'mri' / 'T1.nii.gz'

        mni152_template = nib.load(mni152_template)
        xform_mni152_tk_ras = mni152_template.header.get_vox2ras_tkr()

        mni152_img = nib.load(mni152_template_nii)

        # get electrodes.tsv file from ses 0 in the data/output/ram directory
        electrodes_tsv_file = list(self.subject_dir.rglob("**/*ieeg*/*ses-0*_electrodes.tsv"))[0]
        electrodes = pd.read_csv(electrodes_tsv_file, sep="\t")
        electrodes = electrodes.rename(columns={'name': 'labels', 'x': 'mm_x', 'y': 'mm_y', 'z': 'mm_z'})

        # Convert world coordinates in mm to voxel space
        mm_coords = electrodes[['mm_x', 'mm_y', 'mm_z']]
        
        # MNI152 world coordinates (mm) -> voxel coordinates
        voxel_coords = nib.affines.apply_affine(np.linalg.inv(mni152_img.affine), mm_coords)

        # Voxel coordinates -> FreeSurfer surface RAS coordinates (mm)
        electrodes_homog = np.hstack((voxel_coords, np.ones((voxel_coords.shape[0], 1))))
        electrodes_surfmm = np.round(np.dot(xform_mni152_tk_ras, electrodes_homog.T).T[:, :3], decimals=4)

        # Add transformed coordinates to electrodes dataframe
        electrodes['surfmm_x'] = electrodes_surfmm[:, 0]
        electrodes['surfmm_y'] = electrodes_surfmm[:, 1]
        electrodes['surfmm_z'] = electrodes_surfmm[:, 2]

        electrodes['vox_x'] = (voxel_coords[:, 0]).astype(int)
        electrodes['vox_y'] = (voxel_coords[:, 1]).astype(int)
        electrodes['vox_z'] = (voxel_coords[:, 2]).astype(int)

        drop_columns = ['size', 'group', 'hemisphere', 'type', 'tal.x', 'tal.y', 'tal.z', 'ind.region', 'das.region', 'stein.region']
        electrodes = electrodes.drop(columns=drop_columns)

        return electrodes
    




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
    data_dir = project_root / "data" / "output" / "ram" / "sub-R1010J"
    standardize_ram = StandardizeRAM(openneuro_subject_dir=data_dir, channel_metadata=channel_metadata)
    standardize_ram.curate_interictal_ieeg(start_time=0.0, end_time=135.0)
    standardize_ram.curate_task_ieeg()
    standardize_ram.curate_ieeg_recon(project_root=project_root)
    
#%%

if __name__ == "__main__":
    typer.run(main)

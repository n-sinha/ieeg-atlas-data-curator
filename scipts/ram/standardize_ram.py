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
import subprocess
import os
import time

from pathlib import Path
from curate_pull_ram import clean_ram_metadata

# Set up logging configuration
def setup_logging(log_file_path=None):
    """
    Set up logging with both console and file output.
    
    Args:
        log_file_path (Path, optional): Path to save log file. If None, only console logging.
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file_path is provided
    if log_file_path:
        # Ensure the log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file_path}")
    
    return logger

# Initialize logging (console only for now)
setup_logging()

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
        (self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg_recon').mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory structure standardized for {subject_id} in {self.subject_dir.parent / 'BIDS'}")

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

        subject_id = self.subject_dir.name
        output_dir = self.subject_dir.parent / "BIDS" / subject_id / 'derivatives' / 'ieeg_recon' / 'module2'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        electrodes_tsv_files = list(self.subject_dir.rglob("**/*ieeg*/*ses-0*_electrodes.tsv"))[0]
        electrodes = pd.read_csv(electrodes_tsv_files, sep="\t")
        electrodes = electrodes.filter(items=['name', 'tal.x', 'tal.y', 'tal.z'])
        
        # Create output file for coordinates in MRI space
        output_file = output_dir / 'electrodes_inMRImm.txt'

        # copy /Users/nishant/Dropbox/Sinha/Lab/Research/projects/discover/epilepsy/IEEG-atlas/ieeg-atlas-data-curator/assets/freesurfer/fsaverage/mri/T1.nii.gz to module 2 as ct_to_mri.nii.gz
        shutil.copy(project_root / 'assets' / 'freesurfer' / 'fsaverage' / 'mri' / 'T1.nii.gz', output_dir / 'ct_to_mri.nii.gz')
        logging.info(f"Copied T1.nii.gz to {output_dir / 'ct_to_mri.nii.gz'}")

        # Write coordinates in MRI space format
        with open(output_file, 'w') as f:
            # Write header
            f.write('Coordinates in Destination volume (in mm)\n')
            
            # Write coordinates with proper spacing
            for _, row in electrodes.iterrows():
                coord_line = f"{row['tal.x']:8.4f}  {row['tal.y']:8.4f}  {row['tal.z']:8.4f}\n"
                f.write(coord_line)
        
        logging.info(f"Electrode coordinates saved to {output_file}")

        talariach_t1 = project_root / 'assets' / 'freesurfer' / 'fsaverage' / 'mri' / 'T1.mgz'
        talariach_t1 = nib.load(talariach_t1)
        electrodes_vox = apply_affine(np.linalg.inv(talariach_t1.affine), electrodes[['tal.x', 'tal.y', 'tal.z']])
        electrodes_vox = np.round(electrodes_vox).astype(int)
        
        # Create output file for coordinates in voxel space
        output_file_vox = output_dir / 'electrodes_inMRIvox.txt'

        # Write coordinates in voxel space format
        with open(output_file_vox, 'w') as f:
            # Write header
            f.write('Coordinates in Destination volume (in voxels)\n')
            
            # Write coordinates with proper spacing (matching the format from electrodes_inMRIvox.txt)
            for coord in electrodes_vox:
                coord_line = f"{coord[0]:8.3f}  {coord[1]:8.3f}  {coord[2]:8.3f}\n"
                f.write(coord_line)

        logging.info(f"Electrode coordinates in voxel space saved to {output_file_vox}")

        # export electrode names in module1 
        output_file_module1 = output_dir.parent / 'module1' / 'electrode_names.txt'
        output_file_module1.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_module1, 'w') as f:
            f.write('\n'.join(electrodes['name']))
        logging.info(f"Electrode names saved to {output_file_module1}")

        # export ants registraion from ants_fsaverage_MNI152 to module4
        output_file_module4 = output_dir.parent / 'module4'
        output_file_module4.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(project_root / 'assets' / 'ants_fsaverage_MNI152', output_file_module4, dirs_exist_ok=True)
        logging.info(f"Ants registraion from ants_fsaverage_MNI152 saved to {output_file_module4}")

    def run_ieeg_recon_docker(self, project_root: Path):
        """
        Run iEEG reconstruction modules 3 and 4 using Docker container.
        This method sets up the required input files and runs the Docker container.
        """
        subject_id = self.subject_dir.name
        output_dir = self.subject_dir.parent / "BIDS" / subject_id / 'derivatives'
        
        # Create input directory for Docker container
        input_dir = output_dir / 'docker_input'
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for required files
        t1_path = project_root / 'assets' / 'freesurfer' / 'fsaverage' / 'mri' / 'T1.nii.gz'
        freesurfer_dir = project_root / 'assets' / 'freesurfer' / 'fsaverage'
        
        # Create dummy CT file (copy T1 as CT since CT is not required but may cause errors)
        dummy_ct_path = input_dir / 'CT.nii.gz'
        if not dummy_ct_path.exists():
            shutil.copy2(t1_path, dummy_ct_path)
            logging.info(f"Created dummy CT file: {dummy_ct_path}")
        
        # Create dummy electrodes file
        dummy_electrodes_path = input_dir / 'electrodes.txt'
        if not dummy_electrodes_path.exists():
            with open(dummy_electrodes_path, 'w') as f:
                f.write("# Dummy electrodes file\n")
                f.write("1 0 0 0\n")  # Single dummy electrode
            logging.info(f"Created dummy electrodes file: {dummy_electrodes_path}")
        
        # Copy T1 to input directory
        t1_input_path = input_dir / 'T1.nii.gz'
        if not t1_input_path.exists():
            shutil.copy2(t1_path, t1_input_path)
            logging.info(f"Copied T1 file to: {t1_input_path}")
        
        # Copy freesurfer directory to input directory
        freesurfer_input_dir = input_dir / 'freesurfer'
        if not freesurfer_input_dir.exists():
            shutil.copytree(freesurfer_dir, freesurfer_input_dir)
            logging.info(f"Copied freesurfer directory to: {freesurfer_input_dir}")
        
        # Detect OS and choose containerization method
        import platform
        os_name = platform.system().lower()
        
        if os_name == 'darwin':  # macOS
            # Build Docker command for macOS
            container_cmd = [
                'docker', 'run',
                '-v', f"{input_dir.absolute()}:/data/input",
                '-v', f"{output_dir.absolute()}:/data/output",
                'nishantsinha89/ieeg_recon:latest',
                '--t1', '/data/input/T1.nii.gz',
                '--ct', '/data/input/CT.nii.gz',
                '--elec', '/data/input/electrodes.txt',
                '--freesurfer-dir', '/data/input/freesurfer',
                '--output-dir', '/data/output',
                '--skip-existing',
                '--modules', '3,4'
            ]
            container_type = "Docker"
            
        elif os_name == 'linux':  # Linux
            # Build Singularity command for Linux
            container_cmd = [
                'singularity', 'run',
                '--bind', f"{input_dir.absolute()}:/data/input",
                '--bind', f"{output_dir.absolute()}:/data/output",
                '--pwd', '/app',  # Set working directory to /app where the script is located
                'singularity/ieeg_recon.sif',  # SIF file instead of Docker image
                '--t1', '/data/input/T1.nii.gz',
                '--ct', '/data/input/CT.nii.gz',
                '--elec', '/data/input/electrodes.txt',
                '--freesurfer-dir', '/data/input/freesurfer',
                '--output-dir', '/data/output',
                '--skip-existing',
                '--modules', '3,4'
            ]
            container_type = "Singularity"
            
        else:
            # Fallback to Docker for other OS (Windows, etc.)
            logging.warning(f"Unsupported OS: {os_name}. Defaulting to Docker.")
            container_cmd = [
                'docker', 'run',
                '-v', f"{input_dir.absolute()}:/data/input",
                '-v', f"{output_dir.absolute()}:/data/output",
                'nishantsinha89/ieeg_recon:latest',
                '--t1', '/data/input/T1.nii.gz',
                '--ct', '/data/input/CT.nii.gz',
                '--elec', '/data/input/electrodes.txt',
                '--freesurfer-dir', '/data/input/freesurfer',
                '--output-dir', '/data/output',
                '--skip-existing',
                '--modules', '3,4'
            ]
            container_type = "Docker"
        
        logging.info(f"Detected OS: {os_name}")
        logging.info(f"Running {container_type} command for iEEG reconstruction module 3...")
        logging.info(f"Command: {' '.join(container_cmd)}")
        
        subprocess.run(container_cmd, check=True)
        # delete container input
        shutil.rmtree(input_dir, ignore_errors=True)

        # rename ct_to_mri.nii.gz to T1.nii.gz
        shutil.move(output_dir / 'ieeg_recon' / 'module2' / 'ct_to_mri.nii.gz', output_dir / 'ieeg_recon' / 'module2' / 'T1.nii.gz')
        logging.info(f"Renamed ct_to_mri.nii.gz to T1.nii.gz for accuracy")
        logging.info(f"{container_type} command completed successfully!")


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

    all_subjects = list((project_root / "data" / "output" / "ram").glob("sub-*"))
    all_subjects = [subject for subject in all_subjects if subject.name == "sub-R1229M"]
    for subject_dir in all_subjects:
        # Set up logging for this specific subject
        log_file_path = project_root / 'logs' / f'{subject_dir.name}.log'
        setup_logging(log_file_path)
        
        try:
            # time it start tic toc
            start_time = time.time()
            logging.info(f"Standardizing {subject_dir.name}")
            standardize_ram = StandardizeRAM(openneuro_subject_dir=subject_dir, channel_metadata=channel_metadata)
            # standardize_ram.curate_interictal_ieeg(start_time=0.0, end_time=135.0)
            # standardize_ram.curate_task_ieeg()
            # standardize_ram.curate_ieeg_recon(project_root=project_root)
            standardize_ram.run_ieeg_recon_docker(project_root=project_root)
            end_time = time.time()
            logging.info(f"Standardization of {subject_dir.name} took {end_time - start_time} seconds")
            logging.info(f"Logs saved to: {log_file_path}")
        except Exception as e:
            logging.error(f"Error standardizing {subject_dir.name}: {e}")
            logging.info(f"Error logs saved to: {log_file_path}")
            continue
#%%

if __name__ == "__main__":
    typer.run(main)

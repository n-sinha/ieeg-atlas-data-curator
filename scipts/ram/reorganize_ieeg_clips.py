import pandas as pd
import json
import logging
from pathlib import Path
import shutil

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def reorganize_ieeg_clips_in_place(bids_root: Path):
    """
    Minimally reorganize ieeg-clips folder structure to be BIDS-compliant.
    Works in place within each subject's derivatives/ieeg-clips directory.
    
    Args:
        bids_root: Path to the BIDS dataset root directory
    """
    # Find all subject directories
    subjects = list(bids_root.glob("sub-*"))
    
    if not subjects:
        logging.error(f"No subject directories found in {bids_root}")
        return
    
    logging.info(f"Found {len(subjects)} subjects to process")
    
    for subject_dir in subjects:
        ieeg_clips_dir = subject_dir / 'derivatives' / 'ieeg-clips'
        
        if not ieeg_clips_dir.exists():
            logging.warning(f"ieeg-clips directory not found for {subject_dir.name}")
            continue
            
        logging.info(f"Processing {subject_dir.name}")
        reorganize_subject_ieeg_clips(ieeg_clips_dir, subject_dir.name)

def reorganize_subject_ieeg_clips(ieeg_clips_dir: Path, subject_id: str):
    """
    Reorganize a single subject's ieeg-clips directory in place.
    
    Args:
        ieeg_clips_dir: Path to the ieeg-clips directory
        subject_id: Subject identifier (e.g., sub-R1010J)
    """
    # 1. Rename channel_metadata.tsv to BIDS-compliant name
    old_channel_file = ieeg_clips_dir / 'channel_metadata.tsv'
    if old_channel_file.exists():
        new_channel_file = ieeg_clips_dir / f'{subject_id}_desc-channelmetadata_channels.tsv'
        
        # Rename the file
        old_channel_file.rename(new_channel_file)
        logging.info(f"Renamed channel_metadata.tsv to {new_channel_file.name}")
        
        # Create JSON sidecar
        create_channel_metadata_json(new_channel_file, subject_id)
    else:
        # Check if the BIDS-compliant file already exists and update its JSON sidecar
        bids_channel_file = ieeg_clips_dir / f'{subject_id}_desc-channelmetadata_channels.tsv'
        if bids_channel_file.exists():
            create_channel_metadata_json(bids_channel_file, subject_id)
            logging.info(f"Updated JSON sidecar for {bids_channel_file.name}")
    
    # 2. Move session files to proper ieeg subdirectories
    for item in ieeg_clips_dir.iterdir():
        if item.is_dir() and item.name.startswith('ses-'):
            reorganize_session_directory(item)

def reorganize_session_directory(session_dir: Path):
    """
    Move session files to proper ieeg subdirectory if they're not already there.
    
    Args:
        session_dir: Path to session directory (e.g., ses-interictal, ses-task)
    """
    # Check if there are any files directly in this session directory
    files_in_session = [f for f in session_dir.iterdir() if f.is_file() and f.suffix in ['.edf', '.tsv', '.json']]
    
    if files_in_session:
        # For ses-interictal, files should go to ses-interictal/ses-0/ieeg/
        # For ses-task, files should go to ses-task/ses-0/ieeg/
        if session_dir.name == 'ses-interictal':
            # Create ses-0 subdirectory first
            ses0_dir = session_dir / 'ses-0'
            if not ses0_dir.exists():
                ses0_dir.mkdir()
            
            # Create ieeg subdirectory inside ses-0
            ieeg_dir = ses0_dir / 'ieeg'
            if not ieeg_dir.exists():
                ieeg_dir.mkdir()
            
            # Move all .edf, .tsv, .json files to ses-0/ieeg subdirectory
            for file_path in files_in_session:
                new_location = ieeg_dir / file_path.name
                file_path.rename(new_location)
                logging.info(f"Moved {file_path.name} to ses-0/ieeg/ subdirectory")
                
                # Create JSON sidecar for interictal EDF files
                if file_path.suffix == '.edf' and 'ses-interictal' in str(file_path):
                    create_interictal_edf_json(new_location)
        else:
            # For other session types, use the original logic
            ieeg_dir = session_dir / 'ieeg'
            if not ieeg_dir.exists():
                ieeg_dir.mkdir()
            
            # Move all .edf, .tsv, .json files to ieeg subdirectory
            for file_path in files_in_session:
                new_location = ieeg_dir / file_path.name
                file_path.rename(new_location)
                logging.info(f"Moved {file_path.name} to ieeg/ subdirectory")
    else:
        # Check if there's an empty ieeg directory that was created unnecessarily
        ieeg_dir = session_dir / 'ieeg'
        if ieeg_dir.exists() and not any(ieeg_dir.iterdir()):
            ieeg_dir.rmdir()
            logging.info(f"Removed empty ieeg directory from {session_dir.name}")
        
        # For ses-interictal, also check for empty ses-0/ieeg directory
        if session_dir.name == 'ses-interictal':
            ses0_ieeg_dir = session_dir / 'ses-0' / 'ieeg'
            if ses0_ieeg_dir.exists() and not any(ses0_ieeg_dir.iterdir()):
                ses0_ieeg_dir.rmdir()
                logging.info(f"Removed empty ses-0/ieeg directory from {session_dir.name}")
    
    # For nested session directories (like ses-task/ses-0/), ensure ieeg files are in ieeg subdirectory
    for subdir in session_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('ses-'):
            sub_ieeg_dir = subdir / 'ieeg'
            if sub_ieeg_dir.exists():
                logging.info(f"ieeg subdirectory already exists in {subdir.name}")

def create_channel_metadata_json(channel_file: Path, subject_id: str):
    """
    Create JSON sidecar file for channel metadata.
    
    Args:
        channel_file: Path to the channel TSV file
        subject_id: Subject identifier
    """
    json_file = channel_file.with_suffix('.json')
    
    json_metadata = {
        "Description": "Channel metadata with boolean flags for seizure onset, interictal spikes, noisy channels, brain lesions, early spread, and epileptic status",
        "Sources": [
            f"bids:raw:{subject_id}/ieeg/",
            f"bids::derivatives/ieeg_recon/module4/electrodes2ROI_mni.csv"
        ],
        "Columns": {
            "labels": "Channel labels from electrodes2ROI_mni.csv",
            "seizure_onset": "Boolean flag indicating if channel is in seizure onset zone",
            "interictal_spikes": "Boolean flag indicating if channel has interictal spikes",
            "noisy_channel": "Boolean flag indicating if channel is noisy/bad electrode",
            "brain_lesions": "Boolean flag indicating if channel has brain lesions",
            "early_spread": "Boolean flag indicating if channel shows early seizure spread",
            "epileptic": "Boolean flag indicating if channel is in epileptic network or is a bad channel"
        },
        "Processing": {
            "Pipeline": "ieeg-atlas-data-curator",
            "Version": "0.1.0",
            "Description": "Channel metadata extracted from raw clinical annotations and mapped to standardized electrode coordinates"
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    logging.info(f"Created JSON sidecar: {json_file.name}")

def create_interictal_edf_json(edf_file: Path):
    """
    Create JSON sidecar file for interictal EDF files with timing information.
    
    Args:
        edf_file: Path to the interictal EDF file
    """
    json_file = edf_file.with_suffix('.json')
    
    json_metadata = {
        "Description": "Interictal iEEG data clipped from session 0 task data",
        "Sources": [
            "bids::derivatives/ieeg-clips/ses-task/ses-0/ieeg/"
        ],
        "Processing": {
            "Pipeline": "ieeg-atlas-data-curator",
            "Version": "0.1.0",
            "Description": "Interictal clips extracted from task session data",
            "Parameters": {
                "start_time": 0.0,
                "end_time": 135.0,
                "unit": "seconds"
            }
        },
        "TaskName": "interictal",
        "RecordingType": "continuous",
        "PowerLineFrequency": 60,
        "SoftwareFilters": "n/a"
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_metadata, f, indent=2)
    
    logging.info(f"Created interictal EDF JSON sidecar: {json_file.name}")

def main():
    """Main function to reorganize ieeg-clips structure for BIDS compliance."""
    setup_logging()
    
    # Get the project root (assuming script is in scipts/ram/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    bids_root = project_root / "data" / "output" / "ram" / "BIDS"
    
    if not bids_root.exists():
        logging.error(f"BIDS root directory not found: {bids_root}")
        return
    
    logging.info(f"Starting minimal BIDS reorganization for: {bids_root}")
    reorganize_ieeg_clips_in_place(bids_root)
    logging.info("BIDS reorganization completed!")

if __name__ == "__main__":
    main()

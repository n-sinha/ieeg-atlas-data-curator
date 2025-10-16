import pandas as pd
import typer
import logging
import re

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
class StandardizeRAMChannels:
    def __init__(self, bids_subject_dir: Path | str, channel_metadata: Path | str):            
        if Path(bids_subject_dir).is_dir():
            self.subject_dir = Path(bids_subject_dir)
        else:
            raise FileNotFoundError(f"Subject directory not found: {bids_subject_dir}")
        
        if Path(channel_metadata).is_file():
            self.channel_metadata = clean_ram_metadata(pd.read_csv(channel_metadata))
        else:
            raise FileNotFoundError(f"Channel metadata file not found: {channel_metadata}")
        
    def validate_channel_names(self):
        subject = self.subject_dir.name
        channel_metadata = self.channel_metadata
        channel_metadata = channel_metadata.loc[subject]

        # Get all the channel names as individual strings from the metadata columns
        # First, get all non-null values from the relevant columns
        metadata_columns = ['Seizure Onset Zone', 'Interictal Spikes', 
                           'Bad Electrodes', 'Brain Lesions', 'Early Spread']
        
        # Collect all channel names from metadata (they are comma-separated strings)
        all_metadata_channels = set()
        for col in metadata_columns:
            if pd.notna(channel_metadata[col]) and channel_metadata[col].strip():
                # Split comma-separated values and clean whitespace
                channels_in_col = [ch.strip() for ch in str(channel_metadata[col]).split(',')]
                all_metadata_channels.update(channels_in_col)
        
        # Get all the channel names from the recon file
        recon_file = self.subject_dir / 'derivatives' / 'ieeg_recon' / 'module4' / 'electrodes2ROI_mni.csv'
        recon = pd.read_csv(recon_file)
        recon_channel_names = set(recon['labels'].values.tolist())
        
        logging.info(f"Metadata channels ({len(all_metadata_channels)}): {sorted(all_metadata_channels)}")
        logging.info(f"Recon channels ({len(recon_channel_names)}): {sorted(recon_channel_names)}")

        # Check if all the channel names in the metadata are in the recon file
        missing_from_recon = all_metadata_channels - recon_channel_names
        
        # Validate that less than 30% of the metadata channels are missing from the recon file
        if len(missing_from_recon)/len(all_metadata_channels) > 0.30:
            error_msg = f"VALIDATION FAILED: {len(missing_from_recon)} out of {len(all_metadata_channels)} channels from metadata are missing from recon file: {sorted(missing_from_recon)}"
            logging.error(error_msg)
            return False
        else:
            logging.info("VALIDATION PASSED: More than 70% of the metadata channels are present in recon file")
            return True

    def make_channel_metadata(self):
        pass
        
def main():
    project_root = Path(__file__).parent.parent.parent
    channel_metadata = project_root / "data" / "input" / "ram" / "channel_metadata.csv"

    all_subjects = list((project_root / "data" / "output" / "ram" / "BIDS").glob("sub-*"))

    for subject_dir in all_subjects:
        # Set up logging for this specific subject
        log_file_path = project_root / 'logs_channels' / f'{subject_dir.name}.log'
        setup_logging(log_file_path)

        try:
            logging.info(f"Standardizing channels for {subject_dir.name}")
            standardize_channels = StandardizeRAMChannels(bids_subject_dir=subject_dir, channel_metadata=channel_metadata)
            standardize_channels.validate_channel_names()
            standardize_channels.make_channel_metadata()
            logging.info(f"Standardization of channels for {subject_dir.name} completed")
            logging.info(f"Logs saved to: {log_file_path}")
        except Exception as e:
            logging.error(f"Error standardizing channels for {subject_dir.name}: {e}")
            logging.info(f"Error logs saved to: {log_file_path}")
            continue


#%%
if __name__ == "__main__":
    typer.run(main)
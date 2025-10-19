import pandas as pd
import typer
import logging
from pathlib import Path
from curate_pull_ram import clean_ram_metadata

def setup_logging(log_file_path=None):
    """Set up logging with both console and file output.
    
    Args:
        log_file_path (Path, optional): Path to save log file. If None, only console logging.
    """
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

setup_logging()


class StandardizeRAMChannels:
    """Class to standardize RAM channel metadata and validate channel names."""
    
    def __init__(self, bids_subject_dir: Path | str, channel_metadata: Path | str):
        """Initialize the StandardizeRAMChannels instance.
        
        Args:
            bids_subject_dir: Path to BIDS subject directory
            channel_metadata: Path to channel metadata CSV file
        
        Raises:
            FileNotFoundError: If subject directory or metadata file not found
        """
        if Path(bids_subject_dir).is_dir():
            self.subject_dir = Path(bids_subject_dir)
        else:
            raise FileNotFoundError(f"Subject directory not found: {bids_subject_dir}")
        
        if Path(channel_metadata).is_file():
            self.channel_metadata = clean_ram_metadata(pd.read_csv(channel_metadata))
        else:
            raise FileNotFoundError(f"Channel metadata file not found: {channel_metadata}")
        
    def validate_channel_names(self):
        """Validate that channel names in metadata match those in reconstruction file.
        
        Returns:
            bool: True if validation passes (less than 30% channels missing), False otherwise
        """
        subject = self.subject_dir.name
        channel_metadata = self.channel_metadata
        subject_data = channel_metadata.loc[subject]
        
        if isinstance(subject_data, pd.DataFrame):
            combined_values = {}
            for col in subject_data.columns:
                values = subject_data[col].dropna().astype(str)
                combined_values[col] = ', '.join(values) if not values.empty else None
            channel_metadata = pd.Series(combined_values)
        else:
            channel_metadata = subject_data

        metadata_columns = ['Seizure Onset Zone', 'Interictal Spikes', 
                           'Bad Electrodes', 'Brain Lesions', 'Early Spread']
        
        all_metadata_channels = set()
        for col in metadata_columns:
            if pd.notna(channel_metadata[col]) and channel_metadata[col].strip():
                channels_in_col = [ch.strip().upper() for ch in str(channel_metadata[col]).split(',')]
                all_metadata_channels.update(channels_in_col)
        
        recon_file = self.subject_dir / 'derivatives' / 'ieeg_recon' / 'module4' / 'electrodes2ROI_mni.csv'
        recon = pd.read_csv(recon_file)
        recon_channel_names = set(recon['labels'].values.tolist())
        
        missing_from_recon = all_metadata_channels - recon_channel_names
        
        if len(missing_from_recon)/len(all_metadata_channels) > 0.30:
            error_msg = f"VALIDATION FAILED: {len(missing_from_recon)} out of {len(all_metadata_channels)} ({len(missing_from_recon)/len(all_metadata_channels)*100:.2f}%) channels from metadata are missing from recon file: {sorted(missing_from_recon)}"
            logging.error(error_msg)
            return False
        else:
            logging.info("VALIDATION PASSED: More than 70% of the metadata channels are present in recon file")
            return True

    def make_channel_metadata(self):
        """Create standardized channel metadata TSV file.
        
        Creates a TSV file with channel labels and boolean flags for:
        - seizure_onset: Channels in seizure onset zone
        - interictal_spikes: Channels with interictal spikes  
        - noisy_channel: Bad/noisy electrodes
        - brain_lesions: Channels with brain lesions
        - early_spread: Channels with early seizure spread
        - epileptic: True if channel appears in any of the above categories
        
        Returns:
            pd.DataFrame: The created channel metadata dataframe
        """
        electrodes2ROI_mni = pd.read_csv(
            self.subject_dir / 'derivatives' / 'ieeg_recon' / 'module4' / 'electrodes2ROI_mni.csv'
        )
        
        channel_metadata = pd.DataFrame(columns=[
            'labels', 'seizure_onset', 'interictal_spikes', 'noisy_channel', 
            'brain_lesions', 'early_spread', 'epileptic'
        ])
        channel_metadata['labels'] = electrodes2ROI_mni['labels'].values.tolist()
        
        subject = self.subject_dir.name
        subject_metadata = self.channel_metadata.loc[subject]
        
        if isinstance(subject_metadata, pd.DataFrame):
            combined_values = {}
            for col in subject_metadata.columns:
                values = subject_metadata[col].dropna().astype(str)
                combined_values[col] = ', '.join(values) if not values.empty else None
            subject_metadata = pd.Series(combined_values)
        
        metadata_mapping = {
            'Seizure Onset Zone': 'seizure_onset',
            'Interictal Spikes': 'interictal_spikes', 
            'Bad Electrodes': 'noisy_channel',
            'Brain Lesions': 'brain_lesions',
            'Early Spread': 'early_spread'
        }
        
        for col in ['seizure_onset', 'interictal_spikes', 'noisy_channel', 'brain_lesions', 'early_spread']:
            channel_metadata[col] = False
        
        for metadata_col, df_col in metadata_mapping.items():
            if pd.notna(subject_metadata[metadata_col]) and subject_metadata[metadata_col].strip():
                channels_in_category = [ch.strip().upper() for ch in str(subject_metadata[metadata_col]).split(',')]
                channel_metadata[df_col] = channel_metadata['labels'].str.upper().isin(channels_in_category)
                logging.info(f"Found {len(channels_in_category)} channels in {metadata_col}")

        channel_metadata['epileptic'] = (
            channel_metadata['seizure_onset'] | 
            channel_metadata['interictal_spikes'] | 
            channel_metadata['noisy_channel'] | 
            channel_metadata['brain_lesions'] | 
            channel_metadata['early_spread']
        )
        
        output_dir = self.subject_dir / 'derivatives' / 'ieeg-clips'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'channel_metadata.tsv'
        channel_metadata.to_csv(output_file, sep='\t', index=False)
        logging.info(f"Channel metadata saved to {output_file}")
        return channel_metadata

        
def main():
    """Main function to standardize channel metadata for all RAM subjects."""
    project_root = Path(__file__).parent.parent.parent
    channel_metadata = project_root / "data" / "input" / "ram" / "channel_metadata.csv"
    all_subjects = list((project_root / "data" / "output" / "ram" / "BIDS").glob("sub-*"))

    for subject_dir in all_subjects:
        log_file_path = project_root / 'logs' / 'channels' / f'{subject_dir.name}.log'
        setup_logging(log_file_path)

        try:
            logging.info(f"Processing {subject_dir.name}")
            standardize_channels = StandardizeRAMChannels(
                bids_subject_dir=subject_dir, 
                channel_metadata=channel_metadata
            )
            if standardize_channels.validate_channel_names():
                standardize_channels.make_channel_metadata()
            logging.info(f"Completed {subject_dir.name}")
        except Exception as e:
            logging.error(f"Error processing {subject_dir.name}: {e}")
            continue


if __name__ == "__main__":
    typer.run(main)
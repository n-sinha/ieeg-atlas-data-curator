# %%
import pandas as pd
from aggregateIEEGhup import IEEGData
from typing import Union
import os
import dotenv
import mne
import shutil

#%%
class IEEGDataDownloader(IEEGData):
    """
    A class to handle IEEG data download from IEEG.org, inheriting from IEEGData.
    """
    def __init__(self):
        """Initialize the IEEGDataDownloader class."""
        super().__init__()  # Initialize parent class
        
        # Load sheet information from environment variables
        self.sheet_id = self._get_env_variable('SHEET_ID_METADATA_PENN')
        self.sheet_name = self._get_env_variable('SHEET_NAME_SEIZURE_DATA_PENN')

    def get_seizure_times(self, subjects: Union[pd.DataFrame, list, None] = None):
        """
        Get seizure timing information for specified subjects.

        Args:
            subjects (Union[pd.DataFrame, list, None], optional): Subject identifiers to filter the data.
                - If pd.DataFrame: Must have subject IDs as index (format: 'sub-RIDxxxx')
                - If list: List of subject IDs as strings (format: 'sub-RIDxxxx')
                - If None: Returns data for all subjects
                Default is None.

        Returns:
            pd.DataFrame: DataFrame containing seizure timing information with columns:
                - record_id (index)
                - onset
                - offset
                - source
        """
        # Get seizure metadata from Google Sheet
        metadata = self.get_google_sheet_data(self.sheet_name, self.sheet_id)
        metadata['record_id'] = 'sub-RID' + metadata['record_id'].astype(str).str.zfill(4)

        # Filter for specific subjects if provided
        if subjects is not None:
            if isinstance(subjects, pd.DataFrame):
                subject_ids = subjects.index
            else:
                subject_ids = subjects
            metadata = metadata[metadata['record_id'].isin(subject_ids)]

        # Select relevant columns and set index
        seizures_times = metadata.filter(['record_id', 'onset', 'offset', 'source'])
        seizures_times = seizures_times.set_index('record_id')

        return seizures_times
    
    def get_sleep_times(self, subjects: Union[pd.DataFrame, list, None] = None):
        """
        Get sleep timing information for specified subjects.

        Args:
            subjects (Union[pd.DataFrame, list, None], optional): Subject identifiers to filter the data.
        """

        pass
        
    
    def get_interical_data_cache(self, record_id: str, cache_dir: str, destination_dir: str, dry_run: bool = True):
        """
        Get cached interical data for a given record ID and copy it to the destination directory.

        Args:
            record_id (str): The record ID of the IEEG.org dataset.
            cache_dir (str): The directory containing the cached data.
            destination_dir (str): The directory where data should be copied.
            dry_run (bool, optional): If True, only show what would be copied without actually copying.
                                    Defaults to True for safety.

        Returns:
            list: List of operations that were performed or would be performed (in dry run mode)
        """
        # Get list of all items in the cache directory
        available_subjects = os.listdir(cache_dir)
        
        # Extract the 3-digit subject numbers from folders that start with 'HUP'
        available_subjects = [x[-3:] for x in available_subjects if x.startswith('HUP')]
        
        # Get normative subjects and filter for those with cached data
        sf_ieeg_subjects = self.normative_ieeg_subjects()
        cache_data = sf_ieeg_subjects[sf_ieeg_subjects['hupsubjno'].isin(available_subjects)]
        
        # Filter for specific record if provided
        if record_id:
            cache_data = cache_data[cache_data.index == record_id]
            if len(cache_data) == 0:
                raise ValueError(f"No cached data found for record ID: {record_id}")

        operations = []
        
        if dry_run:
            print("\nAvailable subjects:")
            print("HUP ID\tRecord ID")
            print("-" * 20)
            for idx, row in cache_data.iterrows():
                print(f"HUP{int(row['hupsubjno']):03d}\t{idx}")
            return operations

        for idx, row in cache_data.iterrows():
            # Construct source directory path
            subject_hup = f"HUP{int(row['hupsubjno']):03d}"
            source_path = os.path.join(cache_dir, subject_hup, 'interictal')
            
            if not os.path.exists(source_path):
                print(f"Skipping: Source directory not found - {source_path}")
                continue
                
            # Find all interictal files
            interictal_files = [x for x in os.listdir(source_path) 
                              if x.startswith('interictal_eeg_bipolar_')]
            
            # Create subject directory in destination
            subject_dir = os.path.join(destination_dir, idx)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Process each file
            for file in interictal_files:
                source_file = os.path.join(source_path, file)
                dest_file = os.path.join(subject_dir, file)
                operations.append(f"Copying: {source_file} -> {dest_file}")
                shutil.copy(source_file, dest_file)
            
        for op in operations:
            print(op)
            
        return operations

#%%
if __name__ == "__main__":
    # Initialize the downloader
    ieeg = IEEGDataDownloader()
    
    # Define the base directory where cached data is stored
    # cache_dir = ieeg.root_dir.parent / 'sixth_sense' / 'data'
    cache_dir = ieeg.root_dir / 'data' / 'hup' / 'derivatives' / 'bipolar' 
    cache_subjects = os.listdir(cache_dir)
    # cache_subjects = [x for x in cache_subjects if x.startswith('HUP')]
    # destination_dir = ieeg.root_dir / 'data' / 'hup' / 'sourcedata'

    #%%
    sf_ieeg_subjects = ieeg.normative_ieeg_subjects()
    sf_ieeg_subjects['hupsubjno'] = 'HUP' + sf_ieeg_subjects['hupsubjno'].astype(str).str.zfill(3)
    sf_ieeg_subjects = sf_ieeg_subjects[~sf_ieeg_subjects.index.isin(cache_subjects)]


    sleep_times = pd.read_pickle(ieeg.root_dir / 'data' / 'hup' / 'derivatives' / 'sleep_times.pkl')
    sf_ieeg_subjects = sf_ieeg_subjects[sf_ieeg_subjects['hupsubjno'].isin(sleep_times['name'])]



    
# %%

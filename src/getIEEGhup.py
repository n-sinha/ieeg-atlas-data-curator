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
    
    def get_interical_data_cache(self, record_id: str, cache_dir: str, destination_dir: str):
        """
        Get cached interical data for a given record ID.

        Args:
            record_id (str): The record ID of the IEEG.org dataset.
            cache_dir (str): The directory to save the cached data.
        """

        available_subjects = os.listdir(cache_dir)
        hup_id = [x[-3:] for x in available_subjects if x.startswith('HUP')]

        
        
#%%
if __name__ == "__main__":
    # Get priority groups from aggregateIEEGhup
    ieeg = IEEGDataDownloader()
    sf_ieeg_subjects = ieeg.normative_ieeg_subjects()

    cache_dir = '/Users/nishant/Dropbox/Sinha/Lab/Research/epi_iEEG_focality/sixth_sense/data'
    available_subjects = os.listdir(cache_dir)
    available_subjects = [x[-3:] for x in available_subjects if x.startswith('HUP')]
    cache_data = sf_ieeg_subjects[sf_ieeg_subjects['hupsubjno'].isin(available_subjects)]
    source_dir = cache_dir + '/HUP' + cache_data['hupsubjno'].astype(str).str.zfill(3) + '/interictal'
    destination_dir = ieeg.root_dir / 'data' / 'hup' / 'sourcedata'

#%%
    for s in range(len(source_dir)):
        interictal_files = [x for x in os.listdir(source_dir[s]) if x.startswith('interictal_eeg_bipolar_')]
        # make a directory for each file
        subject_dir = destination_dir / source_dir.index[s]
        os.makedirs(subject_dir, exist_ok=True)
        for file in interictal_files:
            # copy the file to the new directory
            shutil.copy(source_dir[s] + '/' + file, subject_dir)
# %%

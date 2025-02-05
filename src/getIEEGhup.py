import pandas as pd
from aggregateIEEGhup import IEEGData, main as aggregate_main

class SeizureData(IEEGData):
    """
    A class to handle seizure-related data processing, inheriting from IEEGData.
    """
    def __init__(self):
        """Initialize the SeizureData class."""
        super().__init__()  # Initialize parent class
        self.sheet_name_metadata = 'seizure_data'
        self.sheet_id = '1EBSC7lWiSDuGStGoxOP71qns2LkrYQe9'

    def get_seizure_times(self, subjects=None):
        """
        Get seizure timing information for specified subjects.

        Args:
            subjects (pd.DataFrame, optional): DataFrame with subject IDs as index.
                If None, returns data for all subjects.

        Returns:
            pd.DataFrame: DataFrame containing seizure timing information with columns:
                - record_id (index)
                - onset
                - offset
                - source
        """
        # Get seizure metadata from Google Sheet
        metadata = self.get_google_sheet_data(self.sheet_name_metadata, self.sheet_id)
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

def main():
    """
    Main function to process seizure data for priority 1 subjects.
    
    Returns:
        pd.DataFrame: Seizure timing information for priority 1 subjects
    """
    # Get priority groups from aggregateIEEGhup
    priority_groups = aggregate_main()
    
    # Initialize seizure data processing
    seizures = SeizureData()
    
    # Get seizure times for priority 1 subjects
    priority1_subjects = priority_groups['priority1']
    seizure_times = seizures.get_seizure_times(priority1_subjects)
    
    return seizure_times

if __name__ == "__main__":
    seizure_times = main()
    print(f"Found seizure times for {len(seizure_times)} subjects")

#%% 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
from io import StringIO
from dotenv import load_dotenv
from pathlib import Path
from typing import Union

#%%
class IEEGData:
    def __init__(self):
        """Initialize the IEEGData class."""
        self.root_dir = Path(__file__).parent.parent
        self.env_path = self.root_dir / '.env'
        load_dotenv(dotenv_path=self.env_path)
        
        # Initialize REDCap variables as None
        self.redcap_token = None
        self.redcap_report_id = None

    def _get_env_variable(self, var_name: str) -> str:
        """
        Internal method to safely get an environment variable.
        
        Args:
            var_name (str): Name of the environment variable
        
        Returns:
            str: Value of the environment variable
        
        Raises:
            ValueError: If the environment variable is not set
        """
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set. "
                           f"Please check your .env file.")
        return value

    def get_google_sheet_data(self, sheet_name: str, sheet_id: str) -> pd.DataFrame:
        """
        Fetches data from a Google Sheet and returns it as a pandas DataFrame.
        
        Args:
            sheet_name (str): Name of the sheet to read
            sheet_id (str): Google Sheet ID
        
        Returns:
            pd.DataFrame: DataFrame containing the sheet data
        """
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        return pd.read_csv(url)

    def get_redcap_data(self) -> pd.DataFrame:
        """
        Fetches data from REDCap and returns it as a pandas DataFrame.
        
        Args:
            report_id (str, optional): REDCap report ID to fetch. 
                                     If None, uses the ID from environment variables.
        
        Returns:
            pd.DataFrame: DataFrame containing the REDCap data, filtered for 3T protocols
        """
        # Load REDCap credentials only when needed
        if self.redcap_token is None:
            self.redcap_token = self._get_env_variable('REDCAP_TOKEN')
        if self.redcap_report_id is None:
            self.redcap_report_id = self._get_env_variable('REDCAP_REPORT_ID')
            
        data = {
            'token': self.redcap_token,
            'content': 'report',
            'format': 'csv',
            'report_id': self.redcap_report_id,
            'csvDelimiter': '',
            'rawOrLabel': 'label',
            'rawOrLabelHeaders': 'raw',
            'exportCheckboxLabel': 'false',
            'returnFormat': 'csv'
        }
        
        response = requests.post('https://redcap.med.upenn.edu/api/', data=data)
        df = pd.read_csv(StringIO(response.text))
        df['record_id'] = 'sub-RID' + df['record_id'].astype(str).str.zfill(4)
        
        return df.set_index('record_id').sort_index()
    
    def filter_ieeg_atlas_patients(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Filters IEEG Atlas patients from the REDCap metadata.
        Args:
            metadata (pd.DataFrame): REDCap metadata        
        Returns:
            pd.DataFrame: Filtered IEEG Atlas patients
        """
        # Filter metadata for 'modality_ieeg' == 1 and 'ignore'== 0
        metadata = metadata[metadata['modality_ieeg'] == 1]
        metadata = metadata[metadata['ignore'] == 0]
        metadata = metadata[metadata['seizure_Engel12m']<2]
        return metadata
        
    def normative_ieeg_subjects(self) -> pd.DataFrame:
        """
        Cleans and filters patient data from REDCap and metadata sources.
        
        Returns:
            pd.DataFrame: Cleaned REDCap data containing only eligible patients
        """
        # Get initial REDCap data
        metadata_redcap = self.get_redcap_data()
        
        # Clean REDCap data
        columns_to_keep = [
            'hupsubjno', 'ieegportalsubjno', 'intervention_pecclinical',
            'months_at_followup_1', 'engel_class_pecclinical',
            'months_at_followup_2', 'engel_class_2_pecclinical'
        ]
        metadata_redcap = metadata_redcap.filter(columns_to_keep)
        
        # Filter based on clinical criteria
        metadata_redcap = (metadata_redcap[
            metadata_redcap['intervention_pecclinical'].isin(['Resection', 'Laser Ablation'])]
            .dropna(subset=['engel_class_pecclinical'])
            .loc[lambda df: df['engel_class_pecclinical'].astype(str).str.startswith(('IB','IA'))]
            .loc[lambda df: ~df['engel_class_2_pecclinical'].astype(str).str.startswith(('II', 'III', 'IV'))]
        )
        
        # Get and process metadata from Google Sheets for prior lesions in Penn metadata
        sheet_name_metadata = 'metadata'
        sheet_id = '1EBSC7lWiSDuGStGoxOP71qns2LkrYQe9' 
        metadata = self.get_google_sheet_data(sheet_name_metadata, sheet_id)
        
        # Format metadata
        metadata['record_id'] = 'sub-RID' + metadata['record_id'].astype(str).str.zfill(4)
        metadata = (metadata.set_index('record_id')
                   .sort_index()
                   .loc[lambda df: df.index.isin(metadata_redcap.index)]
                   .filter(['ignore', 'ignore_reason']))
        
        # Filter metadata for prior lesions
        metadata['ignore_reason'] = metadata['ignore_reason'].fillna('')
        metadata = metadata[metadata['ignore_reason'].str.startswith('prior lesion')]
        
        # Remove excluded patients from REDCap data
        sf_ieeg_subjects = metadata_redcap[~metadata_redcap.index.isin(metadata.index)]
        
        return sf_ieeg_subjects

    def get_ieeg_recon_status(self, bids_dir: str, cnt_dir: str, subjects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check IEEG reconstruction status for subjects in both BIDS and CNT directories.
        
        Args:
            bids_dir (str): Path to the BIDS directory
            cnt_dir (str): Path to the CNT Implant Reconstructions directory
            subjects_df (pd.DataFrame): DataFrame with subject IDs as index (must have record_ID)
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - ieeg_recon_status: 'processed' (in BIDS), 'available' (in CNT), or 'missing'
                - ieeg_recon_path: Full path to reconstruction data if available
        """
        # Create copy of input DataFrame to avoid modifying original
        result_df = subjects_df.copy()
        
        # Initialize new columns
        result_df['ieeg_recon_status'] = None
        result_df['ieeg_recon_path'] = None
        
        # Check BIDS directory status
        recon_status = self.check_ieeg_recon_status(bids_dir, result_df)
        
        # For subjects not in BIDS, check CNT directory
        cnt_recon_status = self.check_cnt_recon_status(cnt_dir, recon_status['not_processed'])
        
        # Update status and paths
        result_df.loc[recon_status['processed'], 'ieeg_recon_status'] = 'processed'
        result_df.loc[recon_status['processed'], 'ieeg_recon_path'] = [
            os.path.join(bids_dir, subject, 'derivatives/ieeg_recon/module2')
            for subject in recon_status['processed']
        ]
        
        result_df.loc[cnt_recon_status['available'].index, 'ieeg_recon_status'] = 'available'
        result_df.loc[cnt_recon_status['available'].index, 'ieeg_recon_path'] = cnt_recon_status['available']['full_path']
        
        # Mark remaining subjects as missing
        result_df.loc[cnt_recon_status['missing'], 'ieeg_recon_status'] = 'missing'
        
        return result_df

    def get_postopMRI_status(self, bids_dir: str, subjects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check both IEEG reconstruction and postop MRI status for subjects.
        
        Args:
            bids_dir (str): Path to the BIDS directory
            subjects_df (pd.DataFrame): DataFrame with subject IDs as index
        
        Returns:
            pd.DataFrame: Original DataFrame with added column:
                - postop_mri_status: 'available' or 'missing'
        """
        # Create copy of input DataFrame to avoid modifying original
        result_df = subjects_df.copy()
        
        # Get all subjects from BIDS directory
        available_subjects = os.listdir(bids_dir)
        
        # Check postop MRI status
        result_df['postop_mri'] = 'missing'
        for subject in available_subjects:
            if subject in result_df.index:
                postop_path = os.path.join(bids_dir, subject, 'ses-postop01')
                if os.path.exists(postop_path):
                    result_df.loc[subject, 'postop_mri'] = 'available'
        
        return result_df

    def check_ieeg_recon_status(self, bids_dir: str, query_subjects: Union[str, pd.DataFrame, list]) -> dict:
        """
        Check which patients have completed IEEG reconstruction processing.
        
        Args:
            bids_dir (str): Path to the BIDS directory
            query_subjects (Union[str, pd.DataFrame, list]): Subject(s) to query. Can be:
                - A single subject ID string (e.g., 'sub-RID0001')
                - A DataFrame with subject IDs as index
                - A list of subject IDs
        
        Returns:
            dict: Dictionary containing two keys:
                - 'processed': List of subjects with completed IEEG reconstruction
                - 'not_processed': List of subjects without IEEG reconstruction
        """
        # Convert input to list of subject IDs
        if isinstance(query_subjects, pd.DataFrame):
            subjects_to_check = query_subjects.index.tolist()
        elif isinstance(query_subjects, str):
            subjects_to_check = [query_subjects]
        elif isinstance(query_subjects, list):
            subjects_to_check = query_subjects
        else:
            raise ValueError("query_subjects must be a string, DataFrame, or list")

        # Get all processed subjects from BIDS directory
        available_subjects = os.listdir(bids_dir)
        
        # Create paths to ieeg_recon directories
        recon_paths = {
            subject: os.path.join(bids_dir, subject, 'derivatives/ieeg_recon/module2')
            for subject in available_subjects
        }
        
        # Check which paths exist
        processed_subjects = [
            subject for subject, path in recon_paths.items()
            if os.path.exists(path) and subject in subjects_to_check
        ]
        
        # Find subjects without processing
        unprocessed_subjects = [
            subject for subject in subjects_to_check
            if subject not in processed_subjects
        ]
        
        return {
            'processed': processed_subjects,
            'not_processed': unprocessed_subjects
        }

    def check_cnt_recon_status(self, cnt_recon_dir: str, query_subjects: Union[str, pd.DataFrame, list]) -> dict:
        """
        Check which patients have reconstructions in the CNT Implant Reconstructions directory.
        
        Args:
            cnt_recon_dir (str): Path to the CNT Implant Reconstructions directory
            query_subjects (Union[str, pd.DataFrame, list]): Subject(s) to query. Can be:
                - A single subject ID string (e.g., 'sub-RID0001')
                - A DataFrame with subject IDs as index
                - A list of subject IDs
        
        Returns:
            dict: Dictionary containing:
                - 'processed': DataFrame with subject IDs as index and full reconstruction paths
                - 'not_processed': List of subjects without CNT reconstruction
        """
        # Convert input to list of subject IDs
        if isinstance(query_subjects, pd.DataFrame):
            subjects_to_check = query_subjects.index.tolist()
        elif isinstance(query_subjects, str):
            subjects_to_check = [query_subjects]
        elif isinstance(query_subjects, list):
            subjects_to_check = query_subjects
        else:
            raise ValueError("query_subjects must be a string, DataFrame, or list")

        # Get CNT reconstruction folders and extract RIDs
        cnt_folders = os.listdir(cnt_recon_dir)
        recon_data = pd.DataFrame({
            'folder_name': cnt_folders,
            'rid_numbers': [re.findall(r'\d+', folder) for folder in cnt_folders]
        })
        
        # Clean up the data
        recon_data = recon_data[recon_data['rid_numbers'].map(len) > 0]
        recon_data['record_id'] = recon_data['rid_numbers'].map(lambda x: x[0])
        recon_data['record_id'] = 'sub-RID' + recon_data['record_id'].str.zfill(4)
        recon_data['full_path'] = recon_data['folder_name'].map(
            lambda x: os.path.join(cnt_recon_dir, x)
        )
        
        # Set index and keep relevant columns
        recon_data = recon_data.set_index('record_id')[['full_path']]
        
        # Filter for requested subjects
        processed_subjects = recon_data[recon_data.index.isin(subjects_to_check)]
        
        # Find unprocessed subjects
        unprocessed_subjects = [
            subject for subject in subjects_to_check 
            if subject not in processed_subjects.index
        ]
        
        return {
            'available': processed_subjects,
            'missing': unprocessed_subjects
        }

#%%
if __name__ == "__main__":
    ieeg = IEEGData()
    sf_ieeg_subjects = ieeg.normative_ieeg_subjects()
    print(sf_ieeg_subjects['intervention_pecclinical'].value_counts())

    # Use new function to get reconstruction status
    bids_dir = '/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS'
    cnt_dir = '/Users/nishant/Library/CloudStorage/Box-Box/CNT Implant Reconstructions'
    sf_ieeg_subjects = ieeg.get_ieeg_recon_status(bids_dir, cnt_dir, sf_ieeg_subjects)
    sf_ieeg_subjects = ieeg.get_postopMRI_status(bids_dir, sf_ieeg_subjects)

    # Define priority groups
    priority1 = sf_ieeg_subjects[
        (sf_ieeg_subjects['ieeg_recon_status'] == 'processed') & 
        (sf_ieeg_subjects['postop_mri'] == 'available')]
    
    priority2 = sf_ieeg_subjects[
        (sf_ieeg_subjects['ieeg_recon_status'] == 'available') & 
        (sf_ieeg_subjects['postop_mri'] == 'available')]
    
    priority3 = sf_ieeg_subjects[
        (sf_ieeg_subjects['ieeg_recon_status'] == 'processed') & 
        (sf_ieeg_subjects['postop_mri'] == 'missing')]
    
    priority4 = sf_ieeg_subjects[
        (sf_ieeg_subjects['ieeg_recon_status'] == 'available') & 
        (sf_ieeg_subjects['postop_mri'] == 'missing')]
        
    priority5 = sf_ieeg_subjects[
        (sf_ieeg_subjects['ieeg_recon_status'] == 'missing') & 
        (sf_ieeg_subjects['postop_mri'] == 'missing')]
    
    # Print summary statistics
    print(f"\nPriority Group Summary:")
    print(f"Priority 1 (ieeg-recon done and postop MRI available): {len(priority1)}")
    print(f"Priority 2 (ieeg-recon available and postop MRI available): {len(priority2)}")
    print(f"Priority 3 (ieeg-recon done and postop MRI missing): {len(priority3)}")
    print(f"Priority 4 (ieeg-recon available and postop MRI missing): {len(priority4)}")
    print(f"Priority 5 (ieeg-recon missing and postop MRI missing): {len(priority5)}")

    # Create dictionary of priority groups
    priority_dict = {
        'priority1': priority1,
        'priority2': priority2,
        'priority3': priority3,
        'priority4': priority4,
        'priority5': priority5
    }

    # Create output directory if it doesn't exist
    output_dir = ieeg.root_dir / 'data' / 'hup' / 'metadata' / 'priority_groups'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each priority group to a CSV file
    for priority_name, priority_df in priority_dict.items():
        output_path = output_dir / f'{priority_name}.csv'
        priority_df.to_csv(output_path)
        print(f"Saved {priority_name} to {output_path}")

    # Create and save summary DataFrame
    summary_df = pd.DataFrame({
        'Priority Group': [
            'Priority 1 (ieeg-recon done and postop MRI available)',
            'Priority 2 (ieeg-recon available and postop MRI available)',
            'Priority 3 (ieeg-recon done and postop MRI missing)',
            'Priority 4 (ieeg-recon available and postop MRI missing)',
            'Priority 5 (ieeg-recon missing and postop MRI missing)'
        ],
        'Count': [len(df) for df in priority_dict.values()]
    })
    
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

# %%

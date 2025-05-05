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
from IPython import embed

#%%
class MetadataPenn:
    def __init__(self):
        """Initialize the IEEGMetaData class."""

        self.project_root = Path(__file__).parent.parent.parent
        dotenv_path = self.project_root  / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        self.bids_path = Path(os.getenv('BIDS_PATH'))
        
        self.redcap_token = os.getenv('REDCAP_TOKEN')
        self.redcap_report_id = os.getenv('REDCAP_REPORT_ID')

        self.sheet_id_metadata = os.getenv('SHEET_ID_METADATA')
        self.sheet_name_metadata = os.getenv('SHEET_NAME_METADATA')
        
        self.manual_validation = os.getenv('SHEET_ID_MANUAL_VALIDATION')
        self.sheet_name_manual_validation_soz = os.getenv('SHEET_NAME_MANUAL_VALIDATION_SOZ')

    def get_metadata_Penn(self) -> pd.DataFrame:
        """
        Fetches data from a Google Sheet and returns it as a pandas DataFrame.
        """
        # Get metadata from Google Sheet
        sheet_id = self.sheet_id_metadata
        sheet_name = self.sheet_name_metadata
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

        metadata = pd.read_csv(url)
        metadata['record_id'] = 'sub-RID' + metadata['record_id'].astype(str).str.zfill(4)
        metadata = metadata.set_index('record_id').sort_index()

        return metadata
    
    def get_manual_validation_soz(self) -> pd.DataFrame:
        """
        Retrieves and processes seizure onset zone (SOZ) manual validation data.
        
        This method:
        1. Fetches SOZ validation data from a Google Sheet
        2. Cleans the patient names by removing 'HUP' prefix
        3. Retrieves REDCap metadata for cross-referencing
        4. Merges the validation data with REDCap data based on hospital subject numbers
        5. Formats the resulting dataframe for consistency with other metadata
        
        Returns:
            pd.DataFrame: Combined dataframe containing REDCap data merged with 
                         manual SOZ validation information, indexed by record_id
        """
        
        sheet_id = self.manual_validation
        sheet_name = self.sheet_name_manual_validation_soz
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

        manual_validation_soz = pd.read_csv(url)
    
        manual_validation_soz.loc[:, 'name'] = manual_validation_soz.loc[:, 'name'].str.replace('HUP', '')
        metadata_redcap = self.get_redcap_data()

        # Create a mapping from hupsubjno to record_id (index)
        hup_to_record_map = metadata_redcap['hupsubjno'].to_dict()
        record_to_hup_map = {v: k for k, v in hup_to_record_map.items()}
        
        # Merge the dataframes
        redcap_validated = pd.merge(metadata_redcap.reset_index(), 
                          manual_validation_soz, 
                          left_on='hupsubjno', 
                          right_on='name', 
                          how='left')
        
        # Set the index back to record_id
        redcap_validated = redcap_validated.set_index('record_id')
        
        # Drop the 'name' column from manual validation as it's redundant
        if 'name' in redcap_validated.columns:
            redcap_validated = redcap_validated.drop(columns=['name'])
            
        return redcap_validated

    def get_redcap_data(self) -> pd.DataFrame:
        """
        Fetches and processes patient data from the REDCap database.
        
        This method:
        1. Connects to REDCap API using authentication token
        2. Retrieves data based on the configured report ID
        3. Formats record IDs to match project conventions (sub-RID####)
        4. Filters to keep only relevant clinical and demographic columns
        
        Returns:
            pd.DataFrame: Cleaned dataframe containing patient information from REDCap,
                         indexed by record_id with standardized formatting
        """
            
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
        metadata_redcap = pd.read_csv(StringIO(response.text))
        metadata_redcap['record_id'] = 'sub-RID' + metadata_redcap['record_id'].astype(str).str.zfill(4)
        
        metadata_redcap = metadata_redcap.set_index('record_id').sort_index()
        
        # Clean REDCap data
        columns_to_keep = [
            'hupsubjno', 'ieegportalsubjno', 'intervention_pecclinical',
            'months_at_followup_1', 'engel_class_pecclinical',
            'months_at_followup_2', 'engel_class_2_pecclinical'
        ]
        metadata_redcap = metadata_redcap.filter(columns_to_keep)

        return metadata_redcap
        
    def query_metadata(self, subject_id: str) -> pd.DataFrame:
        """
        Retrieves and combines all available metadata for a specific subject.
        
        This method gathers information from multiple sources (REDCap and Penn metadata)
        and combines them into a single comprehensive record for the requested subject.
        
        Args:
            subject_id (str): The subject identifier in format 'sub-RID####'
        
        Returns:
            pd.DataFrame: Combined metadata for the specified subject from all available
                         sources, with each source's data concatenated as a single series
        """

        # Check if surgery mask exists
        surgery_mask = self.bids_path / subject_id / 'derivatives' / 'post_to_pre' / 'surgerySeg_in_preT1.nii.gz'
        surgery_mask_exists = surgery_mask.exists()
        
        # Create a Series for surgery mask
        surgery_data = pd.Series({'surgery_mask': surgery_mask_exists})
        
        # Get other metadata
        metadata_penn = self.get_metadata_Penn()
        metadata_redcap = self.get_manual_validation_soz()

        metadata_redcap_subject = metadata_redcap.loc[subject_id]
        metadata_penn_subject = metadata_penn.loc[subject_id]
        
        # Combine all metadata with surgery mask first
        metadata = pd.concat([metadata_redcap_subject,surgery_data,metadata_penn_subject])

        metadata.name = subject_id
                
        return metadata



#%%
if __name__ == "__main__":
    ieeg = MetadataPenn()
    manual_validation_soz = ieeg.get_manual_validation_soz()
    subjects = [
        'sub-RID0596', 
        'sub-RID0194',
        'sub-RID0502',
        'sub-RID0839',
        'sub-RID0786',
        'sub-RID0646',
        'sub-RID0825']
    
    metadata_list = []
    for subject in subjects:
        metadata = ieeg.query_metadata(subject)
        metadata_list.append(metadata)
        
    metadata_df = pd.concat(metadata_list, axis=1)
    

# %%

#%%
import pandas as pd
import h5py
import re
import numpy as np
from pathlib import Path
from IPython import embed
from multiprocessing import Pool
from typing import Union, List, Tuple
import mne
from scipy import signal
import os
from process_ieeg_utils import IEEGTools

#%%
class MNIClipProcessor():
    def __init__(self, subject_id: str):
        super().__init__()
        self.project_root = Path(__file__).parent.parent
        self.electrodes2ROI = Path(self.project_root, 'data', 'derivatives', 'MNI', subject_id, 'electrodes2ROI_mni152_corrected.csv')
        self.ieeg_clip_wake = Path(self.project_root, 'data', 'derivatives', 'MNI', subject_id, 'ieeg_wake.csv')
        self.ieeg_clip_REM = Path(self.project_root, 'data', 'derivatives', 'MNI', subject_id, 'ieeg_R.csv')
        self.ieeg_clip_N2 = Path(self.project_root, 'data', 'derivatives', 'MNI', subject_id, 'ieeg_N2.csv')
        self.ieeg_clip_N3 = Path(self.project_root, 'data', 'derivatives', 'MNI', subject_id, 'ieeg_N3.csv')

    def save_ieeg_processed(self, subject_id: str) -> None:
        """Save the processed iEEG data and electrode information to a CSV file.
        Args:
            ieeg_filtered (pd.DataFrame): Filtered iEEG data
            sampling_rate (float): Sampling rate of the data
            electrodes2ROI (pd.DataFrame): Electrode information
            subject_id (str): Subject ID
        """
        destination_path = self.electrodes2ROI.parent
        h5_file_path = destination_path / 'interictal_ieeg_processed.h5'

        electrodes2ROI = pd.read_csv(self.electrodes2ROI)

        # Check if file exists and handle accordingly
        if h5_file_path.exists():
            print(f"File already exists at {h5_file_path}. Will overwrite.")

        ieeg_clip_wake = pd.read_csv(self.ieeg_clip_wake, header=None).transpose()
        ieeg_clip_REM = pd.read_csv(self.ieeg_clip_REM, header=None).transpose()
        ieeg_clip_N2 = pd.read_csv(self.ieeg_clip_N2, header=None).transpose()
        ieeg_clip_N3 = pd.read_csv(self.ieeg_clip_N3, header=None).transpose()
        
        with h5py.File(h5_file_path, 'w') as f:
            # Create a group for this subject
            subj_group = f.create_group('bipolar_montage')
            
            # Save iEEG data as float32 to save space
            ieeg_h5 = subj_group.create_dataset('ieeg', 
                                    data=ieeg_clip_wake.values.astype(np.float32),
                                    dtype='float32', 
                                    compression='gzip',
                                    compression_opts=4)   # Optimize for time-series access
            
            # Add metadata as attributes
            ieeg_h5.attrs['sampling_rate'] = 200
            ieeg_h5.attrs['channels_labels'] = electrodes2ROI['labels'].tolist()
            ieeg_h5.attrs['shape'] = ieeg_clip_wake.shape
            ieeg_h5.attrs['subject_id'] = subject_id
            ieeg_h5.attrs['roi'] = electrodes2ROI['roi'].tolist()
            ieeg_h5.attrs['roiNum'] = electrodes2ROI['roiNum'].tolist()
            
            # Save electrode data
            coords_data = electrodes2ROI[['mm_x', 'mm_y', 'mm_z']].values.astype(np.float32)
            mni_coord_mm = subj_group.create_dataset('coordinates', 
                                                data=coords_data,
                                                dtype='float32', 
                                                compression='gzip')
            
            # Add electrode metadata
            mni_coord_mm.attrs['labels'] = electrodes2ROI['labels'].tolist()
            mni_coord_mm.attrs['roi'] = electrodes2ROI['roi'].tolist()
            mni_coord_mm.attrs['roiNum'] = electrodes2ROI['roiNum'].tolist()
            mni_coord_mm.attrs['spared'] = [True] * len(electrodes2ROI)
            
        print(f"Successfully saved processed iEEG data for {subject_id} to {h5_file_path}")

#%%
if __name__ == "__main__":

    mni_subjects = os.listdir('/Users/nishant/Dropbox/Sinha/Lab/Research/ieeg_atlas_harmonization/data/derivatives/MNI')
    mni_subjects = [subject for subject in mni_subjects if subject.startswith('sub-')]

    for subject in mni_subjects:
        mni = MNIClipProcessor(subject)
        mni.save_ieeg_processed(subject)
#%%

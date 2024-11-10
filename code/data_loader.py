# data_loader.py

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import nibabel as nib  # Import nibabel to handle NIfTI files

class AtlasDataLoader:
    def __init__(self, base_path):
        """
        Initialize data loader with base path.
        """
        self.base_path = base_path

    def load_atlas_data(self):
        """Load and prepare atlas data."""
        # Load .mat files
        hup_atlas = sio.loadmat(os.path.join(self.base_path, 'HUP_atlas.mat'))
        mni_atlas = sio.loadmat(os.path.join(self.base_path, 'MNI_atlas.mat'))
        
        # Create basic DataFrames
        self.hup_coords = pd.DataFrame(hup_atlas['mni_coords'], columns=['x', 'y', 'z'])
        self.mni_coords = pd.DataFrame(mni_atlas['ChannelPosition'], columns=['x', 'y', 'z'])
        self.hup_ts = pd.DataFrame(hup_atlas['wake_clip'])
        self.mni_ts = pd.DataFrame(mni_atlas['Data_W'])
        
        # Extract patient information
        self.hup_patient_ids = np.unique(hup_atlas['patient_no'])
        self.mni_patient_ids = np.unique(mni_atlas['Patient'])
        
        # Get sampling frequencies
        self.mni_samp_freq = int(np.nanmax(mni_atlas['SamplingFrequency']))
        self.hup_samp_freq = int(np.nanmax(hup_atlas['SamplingFrequency']))
        
        # Create electrode to patient mappings
        self._create_electrode_mappings(hup_atlas, mni_atlas)
        
        # Load the atlas NIfTI file
        atlas_nii_path = os.path.join(self.base_path, 'AAL.nii.gz')
        atlas_img = nib.load(atlas_nii_path)
        self.atlas_data = {
            'data': atlas_img.get_fdata(),
            'hdr': {'Transform': atlas_img.affine}
        }
        self.atlas_data['data'][self.atlas_data['data'] > 9000] = 0
        
        return self

    def _create_electrode_mappings(self, hup_atlas, mni_atlas):
        """Create mappings between electrodes and patients."""
        # HUP mappings
        hup_patient_numbers = hup_atlas['patient_no'].flatten()
        self.hup_idx_map_arr = np.array([num for num in hup_patient_numbers])
        
        # MNI mappings
        mni_patient_numbers = mni_atlas['Patient'].flatten()
        self.mni_idx_map_arr = np.array([num for num in mni_patient_numbers])

# data_loader.py

import os
import scipy.io as sio
import numpy as np
import pandas as pd
import numbers

class AtlasDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_atlas_data(self):
        """Load and prepare atlas data."""
        # Load .mat files
        hup_atlas = sio.loadmat(os.path.join(self.base_path, 'HUP_atlas.mat'), squeeze_me=True)
        mni_atlas = sio.loadmat(os.path.join(self.base_path, 'MNI_atlas.mat'), squeeze_me=True)

        self.roi_info = pd.read_csv(
            os.path.join(self.base_path, 'roiAAL.csv'),
            sep=','  # Use comma as the delimiter
        )

        # Print type and shape of 'ChannelRegion'
        # print("Type of mni_atlas['ChannelRegion']:", type(mni_atlas['ChannelRegion']))
        # print("Shape of mni_atlas['ChannelRegion']:", mni_atlas['ChannelRegion'].shape)

        # Print the first element to see its structure
        # print("First element of mni_atlas['ChannelRegion']:", mni_atlas['ChannelRegion'][0])
        # print("Type of first element:", type(mni_atlas['ChannelRegion'][0]))

        # Verify columns
        # print("Columns in roi_info:", self.roi_info.columns.tolist())

        # Extract HUP data
        self.hup_coords = pd.DataFrame(hup_atlas['mni_coords'], columns=['x', 'y', 'z'])
        self.hup_ts = pd.DataFrame(hup_atlas['wake_clip'])

        # Extract MNI data
        self.mni_coords = pd.DataFrame(mni_atlas['ChannelPosition'], columns=['x', 'y', 'z'])
        self.mni_ts = pd.DataFrame(mni_atlas['Data_W'])

        # Extract patient information
        self.hup_patient_ids = np.unique(hup_atlas['patient_no'])
        self.mni_patient_ids = np.unique(mni_atlas['Patient'])

        # Get sampling frequencies
        self.mni_samp_freq = int(np.nanmax(mni_atlas['SamplingFrequency']))
        self.hup_samp_freq = int(np.nanmax(hup_atlas['SamplingFrequency']))

        # Create electrode to patient mappings
        self._create_electrode_mappings(hup_atlas, mni_atlas)

        # Extract 'ChannelRegion' from MNI data
        self.mni_regions = self._extract_mni_regions(mni_atlas)

        # Map MNI 'ChannelRegion' indices directly to 'roiNum' (since they correspond to 'Sno')
        self.mni_regions_num = self.mni_regions

        return self

    def _extract_mni_regions(self, mni_atlas):
        """Extract 'ChannelRegion' from MNI data."""
        channel_region_raw = mni_atlas['ChannelRegion']
        mni_regions = []

        for idx, region in enumerate(channel_region_raw):
            # print(f"Index {idx}: region = {region}, type = {type(region)}")
            
            if isinstance(region, np.ndarray):
                # Flatten the region if it's nested
                while isinstance(region, np.ndarray) and region.size > 0:
                    region = region[0]
                
                # Now, region should be a scalar
                if isinstance(region, numbers.Number):
                    mni_regions.append(int(region))
                elif isinstance(region, str):
                    try:
                        mni_regions.append(int(region))
                    except ValueError:
                        mni_regions.append(np.nan)
                else:
                    mni_regions.append(np.nan)
            elif isinstance(region, numbers.Number):
                mni_regions.append(int(region))
            else:
                mni_regions.append(np.nan)
        mni_regions = np.array(mni_regions)
        print("Sample of 'ChannelRegion' values:", mni_regions[:10])
        return mni_regions

    # def _extract_mni_regions(self, mni_atlas):
    #     """Extract 'ChannelRegion' from MNI data."""
    #     channel_region_raw = mni_atlas['ChannelRegion']
    #     mni_regions = []
    #     for region in channel_region_raw:
    #         if isinstance(region, np.ndarray) and region.size > 0:
    #             # Extract the numerical index from the array
    #             mni_regions.append(int(region[0]))
    #         elif isinstance(region, (int, float)):
    #             # If already a numerical value
    #             mni_regions.append(int(region))
    #         else:
    #             mni_regions.append(np.nan)
    #     mni_regions = np.array(mni_regions)
    #     print("Sample of 'ChannelRegion' values:", mni_regions[:10])
    #     return mni_regions

    def _create_electrode_mappings(self, hup_atlas, mni_atlas):
        """Create mappings between electrodes and patients."""
        # HUP mappings
        self.hup_idx_map_arr = hup_atlas['patient_no'].flatten()

        # MNI mappings
        self.mni_idx_map_arr = mni_atlas['Patient'].flatten()

# class AtlasDataLoader:
#     def __init__(self, base_path):
#         """
#         Initialize data loader with base path.
#         """
#         self.base_path = base_path

#     def load_atlas_data(self):
#         """Load and prepare atlas data."""
#         # Load .mat files
#         hup_atlas = sio.loadmat(os.path.join(self.base_path, 'HUP_atlas.mat'))
#         mni_atlas = sio.loadmat(os.path.join(self.base_path, 'MNI_atlas.mat'))
        
#         # Create basic DataFrames
#         self.hup_coords = pd.DataFrame(hup_atlas['mni_coords'], columns=['x', 'y', 'z'])
#         self.mni_coords = pd.DataFrame(mni_atlas['ChannelPosition'], columns=['x', 'y', 'z'])
#         self.hup_ts = pd.DataFrame(hup_atlas['wake_clip'])
#         self.mni_ts = pd.DataFrame(mni_atlas['Data_W'])
        
#         # Extract patient information
#         self.hup_patient_ids = np.unique(hup_atlas['patient_no'])
#         self.mni_patient_ids = np.unique(mni_atlas['Patient'])
        
#         # Get sampling frequencies
#         self.mni_samp_freq = int(np.nanmax(mni_atlas['SamplingFrequency']))
#         self.hup_samp_freq = int(np.nanmax(hup_atlas['SamplingFrequency']))
        
#         # Create electrode to patient mappings
#         self._create_electrode_mappings(hup_atlas, mni_atlas)
        
#         # Load the atlas NIfTI file
#         atlas_nii_path = os.path.join(self.base_path, 'AAL.nii.gz')
#         atlas_img = nib.load(atlas_nii_path)
#         self.atlas_data = {
#             'data': atlas_img.get_fdata(),
#             'hdr': {'Transform': atlas_img.affine}
#         }
#         self.atlas_data['data'][self.atlas_data['data'] > 9000] = 0
        
#         return self

#     def _create_electrode_mappings(self, hup_atlas, mni_atlas):
#         """Create mappings between electrodes and patients."""
#         # HUP mappings
#         hup_patient_numbers = hup_atlas['patient_no'].flatten()
#         self.hup_idx_map_arr = np.array([num for num in hup_patient_numbers])
        
#         # MNI mappings
#         mni_patient_numbers = mni_atlas['Patient'].flatten()
#         self.mni_idx_map_arr = np.array([num for num in mni_patient_numbers])

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class AtlasMapper:
    def __init__(self, atlas_data):
        """
        Initialize the AtlasMapper with atlas data.
        
        Args:
            atlas_data (dict): Dictionary containing 'data' (ROI labels) and 'hdr' (header info).
        """
        self.atlas_data = atlas_data
        self.roi_coords_ras, self.roi_labels = self._prepare_roi_coords()
        
    def _prepare_roi_coords(self):
        """
        Prepare voxel coordinates (in RAS space) for each ROI.
        """
        unique_rois = np.unique(self.atlas_data['data'])
        unique_rois = unique_rois[unique_rois > 0]  # Exclude background

        roi_coords = []
        roi_labels = []
        for roi in unique_rois:
            indices = np.argwhere(self.atlas_data['data'] == roi)
            roi_coords.extend(indices)
            roi_labels.extend([roi]*len(indices))
        
        roi_coords = np.array(roi_coords)
        roi_labels = np.array(roi_labels)

        print(f"Total number of ROI voxels: {len(roi_coords)}")
        print(f"First few ROI labels: {roi_labels[:10]}")
        print(f"First few voxel coordinates before transformation: {roi_coords[:5]}")

        # Transform voxel coordinates to RAS space if necessary
        if 'hdr' in self.atlas_data and 'Transform' in self.atlas_data['hdr']:
            affine_transform = self.atlas_data['hdr']['Transform'].T
            roi_coords_ras = np.dot(
                affine_transform,
                np.hstack([roi_coords, np.ones((len(roi_coords), 1))]).T
            ).T[:, :3]
        else:
            roi_coords_ras = roi_coords  # Assume already in RAS if no transform provided
        
        print(f"First few RAS coordinates after transformation: {roi_coords_ras[:5]}")
        return roi_coords_ras, roi_labels
    
    def map_electrodes_to_rois(self, electrode_coords, patient_nums):
        """
        Map electrodes to the nearest ROI in the atlas.
        
        Args:
            electrode_coords (ndarray): Coordinates of electrodes (N x 3).
            patient_nums (ndarray): Patient numbers corresponding to each electrode (N,).
        
        Returns:
            DataFrame: Mapping of electrodes to ROIs with patient numbers.
        """
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.roi_coords_ras)
        distances, indices = nbrs.kneighbors(electrode_coords)
        roi_nums = self.roi_labels[indices.flatten()]
        
        electrode_to_roi = pd.DataFrame({
            'roiNum': roi_nums,
            'patientNum': patient_nums
        })
        return electrode_to_roi

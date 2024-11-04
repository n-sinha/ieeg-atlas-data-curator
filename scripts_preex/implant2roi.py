import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def implant2roi(atlas, electrode_cords, patient_nums):
    """
    Map electrodes to the nearest region of interest (ROI) in the atlas.

    Parameters:
    - atlas: A dictionary containing 'data' with ROI labels and 'hdr' with header info including transformation matrix.
    - electrode_cords: Array of electrode coordinates.
    - patient_nums: Array of patient numbers corresponding to each electrode.

    Returns:
    - DataFrame with columns for ROI number, atlas ROI, and patient number.
    """

    # Get ROI data and coordinates in voxel space (CRS - Column, Row, Slice)
    unique_rois = np.unique(atlas['data'])
    unique_rois = unique_rois[unique_rois > 0]  # Exclude background or non-ROI voxels

    # Prepare voxel coordinates (CRS) for each ROI
    roi_coords = []
    for roi in unique_rois:
        indices = np.argwhere(atlas['data'] == roi)
        for coord in indices:
            roi_coords.append(np.append(coord, roi))  # Append ROI label to coordinates

    roi_coords = np.array(roi_coords)

    # Transform voxel coordinates to RAS space if necessary
    if 'hdr' in atlas and hasattr(atlas['hdr'], 'Transform'):
        affine_transform = atlas['hdr']['Transform'].T
        roi_coords_ras = np.dot(affine_transform, np.hstack([roi_coords[:, :3], np.ones((len(roi_coords), 1))]).T).T
        roi_coords_ras = roi_coords_ras[:, :3]
    else:
        roi_coords_ras = roi_coords[:, :3]  # Assume already in RAS if no transform provided

    # Use nearest neighbors to find closest ROI for each electrode
    nbrs = NearestNeighbors(n_neighbors=1).fit(roi_coords_ras)
    distances, indices = nbrs.kneighbors(electrode_cords)

    # Map electrodes to ROIs
    electrode_to_roi = {
        'roiNum': roi_coords[indices.flatten(), 3],  # ROI labels
        'atlasROI': roi_coords[indices.flatten(), 3],  # This could be customized if ROI labels need mapping
        'patientNum': patient_nums
    }

    return pd.DataFrame(electrode_to_roi)

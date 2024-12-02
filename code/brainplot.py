# %%
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting as niplot

# %%
def get_node_coords(atlas, atlas_LUT):
    '''
    Extract coordinates of brain regions from a given atlas.
    
    Parameters:
    -----------
    atlas : str
        Path to the atlas file (NIfTI format)
    atlas_LUT : pandas.DataFrame
        Look-up table containing ROI information with at least 'roiNum' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing coordinates and labels for each brain region with columns:
        - roiNum: Region ID number
        - mni_x: x-coordinate in MNI space
        - mni_y: y-coordinate in MNI space
        - mni_z: z-coordinate in MNI space
        - Additional columns from atlas_LUT
    '''
    
    coords, labels = niplot.find_parcellation_cut_coords(atlas, return_label_names=True)

    # Convert labels to int64 for consistent data typing
    labels = np.array(labels, dtype=np.int64)

    # Create DataFrame with coordinates and region numbers
    atlas_coord = pd.DataFrame({
        'roiNum': labels,
        'mni_x': coords[:,0], 
        'mni_y': coords[:,1], 
        'mni_z': coords[:,2]
    })
    
    # Merge with look-up table to add region information
    atlas_coord = pd.merge(atlas_LUT, atlas_coord, 
                          on='roiNum',
                          how='inner')

    return atlas_coord

# %% 
if __name__ == "__main__":

    desikan_killiany = '/Users/nishant/Dropbox/Sinha/Lab/Research/t3_freesurfer/cvs_avg35_inMNI152/mri/aparc+aseg.nii.gz'
    atlas_LUT = pd.read_csv('/Users/nishant/Dropbox/Sinha/Lab/Research/dwi_preprocess/atlas_lookuptable/desikanKilliany.csv')
    atlas_coord = get_node_coords(atlas=desikan_killiany, atlas_LUT=atlas_LUT)
    
    # Create interactive 3D visualization
    view = niplot.view_markers(
        atlas_coord[['mni_x', 'mni_y', 'mni_z']],  # Coordinates for each point
        marker_color='red',  # Set marker color
        marker_size=10,  # Set marker size
        marker_labels=atlas_coord['abvr'].tolist()  # Add region abbreviations as labels
    )
    view.open_in_browser()  # Display the interactive plot in web browser

    # Create static visualization
    # np.ones(82) creates uniform sizes for all 82 markers
    niplot.plot_markers(
        np.ones(82),  # Marker sizes (uniform)
        atlas_coord[['mni_x', 'mni_y', 'mni_z']]  # Coordinates for each point
    )

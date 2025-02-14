#%%
import re
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import distance

#%%
class IEEGTools:
    def __init__(self):
        """
        Initialize the IEEGTools class
        """
        pass

    def clean_labels(self, channel_li):
        '''
        This function cleans a list of channels and returns the new channels
        '''
        new_channels = []
        keep_channels = np.ones(len(channel_li), dtype=bool)
        for i in channel_li:
            # standardizes channel names
            M = re.match(r"(\D+)(\d+)", i)

            # account for channels that don't have number e.g. "EKG", "Cz"
            if M is None:
                M = re.match(r"(\D+)", i)
                lead = M.group(1).replace("EEG", "").strip()
                contact = 0
            else:
                lead = M.group(1).replace("EEG", "").strip()
                contact = int(M.group(2))
     
            new_channels.append(f"{lead}{contact:02d}")

        return new_channels
    
    def channels_in_mask(self, ieeg_coords, mask_path, plot=False):
        '''
        This function returns the channels that are in the mask
        
        Parameters:
        -----------
        ieeg_coords : pandas DataFrame
            DataFrame containing electrode coordinates and ROI information
        mask_path : str
            Path to the mask file
        plot : bool, optional
            Whether to create a 3D visualization (default: False)
        '''
        mask = nib.load(mask_path)
        hdr = mask.header
        affine = mask.affine
        mask_data = np.asarray(mask.dataobj)
        # get coordinates of all non-zero voxels in the mask
        mask_coords = np.column_stack(np.where(mask_data != 0))
        mask_coords_mm = apply_affine(affine, mask_coords)
        mask_coords_mm = pd.DataFrame(mask_coords_mm, columns=['x', 'y', 'z'])

        dist = distance.cdist(ieeg_coords.loc[:, ['x','y','z']], mask_coords_mm, 'euclidean')
        dist = np.sum(dist < 5, axis=1) == 0
        ieeg_coords['spared'] = dist

        if plot:
            # plot 3d scatter of mask_coords_mm
            fig = px.scatter_3d(mask_coords_mm, x='x', y='y', z='z', opacity=0.5)
            # add ieeg_coords to the plot and color by binary in_mask
            fig.add_trace(go.Scatter3d(x=ieeg_coords['x'], 
                                    y=ieeg_coords['y'], 
                                    z=ieeg_coords['z'], 
                                    mode='markers',
                                    hovertext=ieeg_coords['roi'],  # Add ROI information to hover
                                    hoverinfo='text',  # Show the hover text
                                    marker=dict(size=5, 
                                              color=ieeg_coords['spared'].astype(int),
                                              colorscale=[[0, 'red'], [1, 'blue']])))  # Custom binary colorscale
            fig.show()        
        return ieeg_coords

if __name__ == "__main__":
    pass

# %%
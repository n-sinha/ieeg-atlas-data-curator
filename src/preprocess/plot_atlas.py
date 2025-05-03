#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from IPython import embed
import h5py
import nibabel as nib
from nilearn import plotting as niplot
from nibabel.freesurfer.io import read_annot
from nibabel.affines import apply_affine
from process_ieeg import IEEGClipProcessor

#%% 
class ProcessSites(IEEGClipProcessor):
    def __init__(self, site_name: str, atlas: Path, atlas_lut: Path):
        super().__init__()
        self.site_name = site_name
        self.atlas = atlas
        self.atlas_lut = atlas_lut
        # check if figure directory exists
        if not self.project_root.joinpath('figures').exists():
            self.project_root.joinpath('figures').mkdir(parents=True, exist_ok=True)
        
    def get_channels(self, epoch_name: str, plot_spared: bool = True):

        all_epoch = list((self.project_root / 'data' / 'derivatives' / self.site_name).rglob(f'**/*{epoch_name}'))
        all_epoch = sorted(all_epoch)
        
        ieeg_channels = pd.DataFrame()
        for clip in all_epoch:
            with h5py.File(clip, 'r') as f:
                f = f['bipolar_montage/coordinates']
                coordinates = pd.DataFrame(f, columns=['x', 'y', 'z'])
                coordinates['labels'] = f.attrs.get('labels')
                coordinates['spared'] = f.attrs.get('spared')
                coordinates['roi'] = f.attrs.get('roi')
                coordinates['roiNum'] = f.attrs.get('roiNum')

                if self.site_name == 'Penn':
                    # get the mni coordinates
                    subject_id = [x for x in clip.parts if x.startswith('sub-')][0]
                    _,_,coordinates_mni152 = self.find_subject_files(subject_id)
                    coordinates_mni152 = pd.read_csv(coordinates_mni152)
                    original_labels = f.attrs.get('original_labels')
                    # find the original labels in the coordinates_mni152 in the same as the original labels
                    coordinates_mni152 = coordinates_mni152[coordinates_mni152['labels'].isin(original_labels)]
                    coordinates_mni152 = coordinates_mni152[['mm_x', 'mm_y', 'mm_z']].reset_index(drop=True)
                    coordinates = pd.concat([coordinates, coordinates_mni152], axis=1)
                    # drop x, y, z from coordinates and rename mm_x, mm_y, mm_z to x, y, z at the same position
                    coordinates = coordinates.drop(columns=['x', 'y', 'z'])
                    coordinates = coordinates.rename(columns={'mm_x': 'x', 'mm_y': 'y', 'mm_z': 'z'})
                    # drop rows where roiNum is None i.e. white matter
                    coordinates = coordinates[coordinates['roiNum'].notna()]
                
                ieeg_channels = pd.concat([ieeg_channels, coordinates])

        ieeg_channels = ieeg_channels.reset_index(drop=True)
        df_mm = ieeg_channels.filter(['x', 'y', 'z'])
        df_surf = self._mm2surf(df_mm, self.atlas)
        df_surf['colors'] = 1
        df_surf['size'] = 1
        df_surf['roi'] = ieeg_channels['roi']
        
        if plot_spared:
            df_surf = df_surf[ieeg_channels.spared]
            df_surf.to_csv(f'{self.project_root}/figures/{self.site_name}.node', sep=' ', index=False, header=False)
        else:
            df_surf = df_surf[~ieeg_channels.spared]
            df_surf.to_csv(f'{self.project_root}/figures/{self.site_name}_arb.node', sep=' ', index=False, header=False)

        return ieeg_channels
    
    def _mm2surf(self, mm_coords: pd.DataFrame, atlas: Path):
        atlas = nib.load(atlas)
        vox_coords = apply_affine(np.linalg.inv(atlas.affine), mm_coords.values)
        vox_coords_hom = np.hstack([vox_coords, np.ones((vox_coords.shape[0], 1))])
        trk_ras = atlas.header.get_vox2ras_tkr()
        surf_coords = np.dot(trk_ras, vox_coords_hom.T).T
        surf_coords = surf_coords[:, :3]
        surf_coords = pd.DataFrame(surf_coords, columns=['x', 'y', 'z'])
        return surf_coords
    
    # def plot_channels_roi(self, channels: pd.DataFrame, plot_spared: bool = True):
        """
        Plot the electrode counts by ROI.
        
        Args:
            channels: DataFrame containing electrode information
            plot_spared: If True, plot spared channels; if False, plot removed channels
        """
        # Filter channels based on the plot_spared parameter
        if plot_spared:
            filtered_channels = channels[channels['spared']]
        else:
            filtered_channels = channels[~channels['spared']]
        
        # Group by ROI and count electrodes
        tbl = filtered_channels.groupby('roi').size().rename('electrode_count')
        
        # Sort by electrode count
        sorted_tbl = tbl.sort_values(ascending=False)
        
        # Make a bar plot with the electrode counts
        plt.figure(figsize=(10, 6), dpi=300)
        sns.barplot(x=sorted_tbl.index, y=sorted_tbl.values)
        plt.xticks(rotation=90)
        plt.xlabel('ROI')
        plt.ylabel('Electrode Count')
        title_type = "Spared" if plot_spared else "Removed"
        plt.title(f'{title_type} Electrode Counts by ROI - {self.site_name}')
        plt.tight_layout()
        
        # Save the figure
        spared_str = "spared" if plot_spared else "removed"
        plt.savefig(f'{self.project_root}/figures/{self.site_name}_{spared_str}_electrode_counts.png')
        
        return sorted_tbl
    
    def plot_all_sites(self, ieeg_channels: pd.DataFrame, atlas_lut: Path, plot_spared: bool = True):
        """
        Create stacked bar plots showing electrode distribution across ROIs for different sites.
        
        Args:
            ieeg_channels: DataFrame containing electrode information from all sites
            atlas_lut: Path to the atlas lookup table CSV file
            plot_spared: If True, plot spared channels; if False, plot removed channels
        """

        if plot_spared:
            channels = ieeg_channels[ieeg_channels['spared']]
        else:
            channels = ieeg_channels[~ieeg_channels['spared']]

        atlas_lut = pd.read_csv(atlas_lut)

        # Sort atlas_lut by isSideLeft and lobe
        atlas_lut = atlas_lut.sort_values(by=['isSideLeft', 'lobe'], ascending=False)
        atlas_lut = atlas_lut.reset_index(drop=True)

        sites = ieeg_channels['site'].unique()

        # Count electrodes for each ROI and site
        for index, row in atlas_lut.iterrows():
            roi = row['roi']
            for site in sites:
                channels_site = channels[channels['site'] == site]
                atlas_lut.loc[index, f'{site}_count'] = channels_site[channels_site['roi'] == roi].shape[0]

        leftROI = atlas_lut.filter(items=['roi', 'lobe', 'MNI_count', 'Penn_count'])[atlas_lut['isSideLeft']==1]
        rightROI = atlas_lut.filter(items=['roi', 'lobe', 'MNI_count', 'Penn_count'])[atlas_lut['isSideLeft']==0]
        
        # Create separate figures for left and right hemispheres
        if plot_spared:
            spared_str = "normative"
        else:
            spared_str = "epileptogenic"
        self._plot_hemisphere(leftROI, "Left Hemisphere", spared_str=spared_str)
        self._plot_hemisphere(rightROI, "Right Hemisphere", spared_str=spared_str)
        
        # Create a combined plot with both hemispheres, grouped by lobe
        self._plot_by_lobe(atlas_lut, sites, spared_str)
    
    def _plot_hemisphere(self, roi_data, hemisphere_name, spared_str="normative"):
        """Helper method to plot stacked bar chart for one hemisphere"""
        # Create data for plotting
        x = roi_data['roi'].values
        y1 = roi_data['MNI_count'].values
        y2 = roi_data['Penn_count'].values
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 8), dpi=300)
        
        # Plot bars
        bars1 = plt.bar(x, y1, color='royalblue', label='MNI')
        bars2 = plt.bar(x, y2, bottom=y1, color='lightcoral', label='Penn')
        
        # Add labels and title
        plt.xlabel('Region of Interest (ROI)', fontsize=12)
        plt.ylabel('Channel Count', fontsize=12)
        plt.title(f'{spared_str} IEEG Contact Distribution by ROI - {hemisphere_name}', fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.legend()
        
        # Add value labels on the bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if y1[i] > 0:
                plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2, 
                        int(y1[i]), ha='center', va='center', color='white', fontsize=8)
            if y2[i] > 0:
                plt.text(bar2.get_x() + bar2.get_width()/2, y1[i] + bar2.get_height()/2, 
                        int(y2[i]), ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.project_root}/figures/{spared_str}_contact_distribution_{hemisphere_name.replace(" ", "_").lower()}.png')
    
    def _plot_by_lobe(self, atlas_lut, sites, spared_str="normative"):
        """Helper method to create a plot grouped by lobes for better organization"""
        # Group by lobe and calculate totals
        lobe_summary = {}
        for lobe in atlas_lut['lobe'].unique():
            lobe_data = atlas_lut[atlas_lut['lobe'] == lobe]
            for site in sites:
                site_col = f'{site}_count'
                if lobe not in lobe_summary:
                    lobe_summary[lobe] = {}
                lobe_summary[lobe][site] = lobe_data[site_col].sum()
        
        # Create DataFrame from summary
        lobe_df = pd.DataFrame(lobe_summary).T
        
        # Plot
        plt.figure(figsize=(10, 6), dpi=300)
        lobe_df.plot(kind='bar', stacked=True, ax=plt.gca(), 
                    color=['royalblue', 'lightcoral'])
        
        plt.xlabel('Brain Lobe', fontsize=12)
        plt.ylabel('Channel Count', fontsize=12)
        plt.title(f'{spared_str} IEEG Contact Distribution by Lobe and Site', fontsize=14)
        plt.legend(title='Site')
        
        # Add count labels
        for i, p in enumerate(plt.gca().patches):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            if height > 0:
                plt.text(x+width/2, y+height/2, int(height), 
                        ha='center', va='center', color='white')
        
        plt.tight_layout()
        plt.savefig(f'{self.project_root}/figures/{spared_str}_contact_distribution_by_lobe.png')

#%% 

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    atlas = Path(project_root, 'data', 'subjects', 'cvs_avg35_inMNI152', 'mri', 'aparc+aseg.mgz')
    atlas_lut = Path(project_root, 'data', 'subjects', 'atlas_lookuptable', 'desikanKilliany.csv')
    epoch = 'interictal_ieeg_processed.h5'

    site_name = 'MNI'
    mni = ProcessSites(site_name, atlas, atlas_lut)
    ieeg_channels_mni = mni.get_channels(epoch_name = epoch, plot_spared=True)

    site_name = 'Penn'
    penn = ProcessSites(site_name, atlas, atlas_lut)
    ieeg_channels_penn = penn.get_channels(epoch_name = epoch, plot_spared=True)
    penn.get_channels(epoch_name = epoch, plot_spared=False)

    ieeg_channels_mni['site'] = 'MNI'
    ieeg_channels_penn['site'] = 'Penn'
    ieeg_channels = pd.concat([ieeg_channels_mni, ieeg_channels_penn], ignore_index=True)

    penn.plot_all_sites(ieeg_channels, atlas_lut)
    penn.plot_all_sites(ieeg_channels, atlas_lut, plot_spared=False)  # Plot removed channels

    
# %%
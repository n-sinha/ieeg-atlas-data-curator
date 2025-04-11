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
from nibabel.freesurfer.io import read_annot
from process_ieeg import IEEGClipProcessor

#%% 

class PlotCoordinates(IEEGClipProcessor):
    def __init__(self, mni_dir: Path, hup_dir: Path, hup_subjects: list):
        super().__init__()
        self.mni_dir = mni_dir
        self.hup_dir = hup_dir
        self.electrodes2ROI_mni = list(self.mni_dir.rglob('**/electrodes2ROI_mni152_corrected.csv'))
        self.electrodes2ROI_hup = list(self.hup_dir.rglob('**/electrodes2ROI_mni152_corrected.csv'))
        self.electrodes2ROI_hup = [f for f in self.electrodes2ROI_hup if f.parts[-5] in hup_subjects] 

    def plot_coordinates(self, dk_parc_lh_annot: Path, dk_parc_rh_annot: Path):
        mni_df = pd.DataFrame()
        for f in self.electrodes2ROI_mni:
            electrodes2ROI = pd.read_csv(f)
            electrodes2ROI['mni_subject'] = f.parts[-5]
            mni_df = pd.concat([mni_df, electrodes2ROI])
        
        hup_df = pd.DataFrame()
        for f in self.electrodes2ROI_hup:
            electrodes2ROI = pd.read_csv(f)
            electrodes2ROI['hup_subject'] = f.parts[-5]
            hup_df = pd.concat([hup_df, electrodes2ROI])

        mni = mni_df.filter(['surfmm_x', 'surfmm_y', 'surfmm_z'])
        mni['colors']= 1
        mni['size']= 1
        mni['labels']= mni_df['roi']
        mni.to_csv('figures/mni.node', sep=' ', index=False, header=False)

        hup = hup_df.filter(['surfmm_x', 'surfmm_y', 'surfmm_z'])
        hup['colors']= 1
        hup['size']= 1
        hup['labels']= hup_df['roi']
        hup.to_csv('figures/hup.node', sep=' ', index=False, header=False)

        lh_annot, lh_ctab, lh_names = read_annot(dk_parc_lh_annot)
        rh_annot, rh_ctab, rh_names = read_annot(dk_parc_rh_annot)

        embed()

        return mni_df, hup_df
    
    def roi_iEEG(self, file_name: str):

        ieeg_files_mni = list(self.mni_dir.rglob(f'**/{file_name}'))
        roi_df_mni = pd.DataFrame()
        for ieeg_file_mni in ieeg_files_mni:
      
            with h5py.File(ieeg_file_mni, 'r') as f:
                f = f['bipolar_montage/coordinates']
                roi = f.attrs.get('roi')
                spared = f.attrs.get('spared')
                labels = f.attrs.get('labels')
                spared_roi = roi[spared]
                spared_labels = labels[spared]
                spared_df = pd.DataFrame(spared_roi, index=spared_labels, columns=['roi'])
                roi_df_mni = pd.concat([roi_df_mni, spared_df], axis=0)

        # Count electrodes per ROI
        electrode_counts_mni = roi_df_mni.groupby('roi').size().rename('electrode_count')
        roi_df_mni = roi_df_mni.groupby('roi').mean()
        # Add the electrode counts to your results
        roi_df_mni['electrode_count'] = electrode_counts_mni

        # get the hup roi_df
        ieeg_files_hup = list(self.hup_dir.rglob(f'**/{file_name}'))
        roi_df_hup = pd.DataFrame()
        
        for ieeg_file_hup in ieeg_files_hup:
            with h5py.File(ieeg_file_hup, 'r') as f:
                f = f['bipolar_montage/coordinates']
                roi = f.attrs.get('roi')    
                labels = f.attrs.get('labels')
                spared = f.attrs.get('spared')
                spared_roi = roi[spared]
                spared_labels = labels[spared]
                spared_df = pd.DataFrame(spared_roi, index=spared_labels, columns=['roi'])
                roi_df_hup = pd.concat([roi_df_hup, spared_df], axis=0)

        electrode_counts_hup = roi_df_hup.groupby('roi').size().rename('electrode_count')
        roi_df_hup = roi_df_hup.groupby('roi').mean()
        roi_df_hup['electrode_count'] = electrode_counts_hup

        return roi_df_mni, roi_df_hup              
        

#%% 

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    mni_dir = Path(project_root, 'data', 'derivatives', 'MNI')

    # TODO: get this in bipolar
    hup_dir = Path('/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS')
    hup_subjects = [
        'sub-RID0031', 'sub-RID0032', 'sub-RID0033', 'sub-RID0050', 'sub-RID0051',
        'sub-RID0064', 'sub-RID0089', 'sub-RID0101', 'sub-RID0117', 'sub-RID0143',
        'sub-RID0167', 'sub-RID0175', 'sub-RID0179', 'sub-RID0190', 'sub-RID0193',
        'sub-RID0222', 'sub-RID0238', 'sub-RID0267', 'sub-RID0301', 'sub-RID0320',
        'sub-RID0322', 'sub-RID0332', 'sub-RID0381', 'sub-RID0405', 'sub-RID0412',
        'sub-RID0424', 'sub-RID0508', 'sub-RID0562', 'sub-RID0589', 'sub-RID0595',
        'sub-RID0621', 'sub-RID0658', 'sub-RID0675', 'sub-RID0679', 'sub-RID0700',
        'sub-RID0785', 'sub-RID0796', 'sub-RID0852', 'sub-RID0883', 'sub-RID0893',
        'sub-RID0941', 'sub-RID0967'
    ]
    plot = PlotCoordinates(mni_dir, hup_dir, hup_subjects)
    file_name = 'interictal_ieeg_processed.h5'
    roi_df_mni, roi_df_hup = plot.roi_iEEG(file_name)
    embed()
# %%
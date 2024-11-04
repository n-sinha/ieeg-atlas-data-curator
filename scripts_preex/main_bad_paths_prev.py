import numpy as np
import nibabel as nib
import pandas as pd

# Import  custom modules
from mergeROIs import merge_rois
from mainUnivar import main_univar
from univariateAbr import univariate_abr
from mainNetwork import main_network
from networkAbr import network_abr
from edgeslist2AbrConn import edgeslist_to_abr_conn
from nodeAbrEdge import node_abr_edge
from implant2roi import implant2roi

from make_seizure_free_abr import make_seizure_free_abr
from make_seizure_free import make_seizure_free


# Set up paths
# data_path = '../Data/'
data_path = '/Users/tereza/nishant/atlas/epi_iEEG_atlas/Data'

# Load a parcellation scheme and metaData
atlas = {}
atlas['data'] = nib.load(f'{data_path}AAL.nii.gz').get_fdata()
atlas['data'][atlas['data'] > 9000] = 0
atlas['hdr'] = nib.load(f'{data_path}AAL.nii.gz').header
roiAAL = pd.read_csv(f'{data_path}roiAAL.mat', sep='\t')  # adjust loading based on actual file format
roiAAL = roiAAL.iloc[:90, :]
customAAL = pd.read_excel(f'{data_path}custom_atlas.xlsx')
atlas['tbl'] = roiAAL
atlasCustom, roiAALcustom = merge_rois(customAAL, roiAAL, atlas)
atlasCustom['tbl'] = roiAALcustom
atlas = atlasCustom

# Load data
MNI_atlas = np.load(f'{data_path}MNI_atlas_orig.npy', allow_pickle=True).item()  # adjust based on actual file format
metaData = pd.read_csv(f'{data_path}metaData.csv')  # adjust loading based on actual file format
HUP_atlasAll = np.load(f'{data_path}HUP_atlas.npy', allow_pickle=True).item()  # adjust based on actual file format
EngelSFThres = 1.1
spikeThresh = 24  # empirical, 1 spike/hour

# Univariate normative modeling of HUP-MNI
norm_MNI_HUP_Atlas, iEEGhupAll, abrnormHUPAtlas = main_univar(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)
iEEGhupAll_z = univariate_abr(norm_MNI_HUP_Atlas, iEEGhupAll)

# Multivariate normative modeling of HUP-MNI
norm_Connection, norm_ConnectionCount, pat_Connection, pat_ConnectionCount = main_network(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)

try:
    pat_Connection = np.load(f'{data_path}pat_ConnectionRedAAL_z.npy', allow_pickle=True).item()
except FileNotFoundError:
    pat_Connection = network_abr(norm_Connection, pat_Connection)

abrConn = edgeslist_to_abr_conn(pat_Connection, HUP_atlasAll)  # Converts edge list to connectivity matrix
percentile_thres = 75
iEEGhupAll_z = node_abr_edge(abrConn, iEEGhupAll_z, percentile_thres)


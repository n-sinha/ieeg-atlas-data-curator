import numpy as np
import nibabel as nib
import pandas as pd
import os

# Import custom modules
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
data_path = '/Users/tereza/nishant/atlas/epi_iEEG_atlas/Data'

# Correct path usage with os.path.join for all files
aal_nii_path = os.path.join(data_path, 'AAL.nii.gz')
# roi_aal_mat_path = os.path.join(data_path, 'roiAAL.mat')
roi_aal_csv_path = os.path.join(data_path, 'roiAAL.csv')
custom_atlas_xlsx_path = os.path.join(data_path, 'custom_atlas.xlsx')
mni_atlas_npy_path = os.path.join(data_path, 'MNI_atlas_orig.npy')
metadata_csv_path = os.path.join(data_path, 'metaData.csv')
hup_atlas_npy_path = os.path.join(data_path, 'HUP_atlas.npy')
pat_conn_red_aal_npy_path = os.path.join(data_path, 'pat_ConnectionRedAAL_z.npy')

# Load a parcellation scheme and metaData
atlas = {
    'data': nib.load(aal_nii_path).get_fdata(),
    'hdr': nib.load(aal_nii_path).header
}
atlas['data'][atlas['data'] > 9000] = 0
# roiAAL = pd.read_csv(roi_aal_mat_path, sep='\t')
# roiAAL = roiAAL.iloc[:90, :]
roiAAL = pd.read_csv(roi_aal_csv_path, sep=',')  # Use ',' for CSV
roiAAL = roiAAL.iloc[:90, :]
customAAL = pd.read_excel(custom_atlas_xlsx_path)
atlas['tbl'] = roiAAL
atlasCustom, roiAALcustom = merge_rois(customAAL, roiAAL, atlas)
atlasCustom['tbl'] = roiAALcustom
atlas = atlasCustom

# Load data
MNI_atlas = np.load(mni_atlas_npy_path, allow_pickle=True).item()
metaData = pd.read_csv(metadata_csv_path)
HUP_atlasAll = np.load(hup_atlas_npy_path, allow_pickle=True).item()
EngelSFThres = 1.1
spikeThresh = 24  # empirical, 1 spike/hour

# Univariate normative modeling of HUP-MNI
norm_MNI_HUP_Atlas, iEEGhupAll, abrnormHUPAtlas = main_univar(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)
iEEGhupAll_z = univariate_abr(norm_MNI_HUP_Atlas, iEEGhupAll)

# Multivariate normative modeling of HUP-MNI
norm_Connection, norm_ConnectionCount, pat_Connection, pat_ConnectionCount = main_network(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)

try:
    pat_Connection = np.load(pat_conn_red_aal_npy_path, allow_pickle=True).item()
except FileNotFoundError:
    print(f"File not found: {pat_conn_red_aal_npy_path}")
    pat_Connection = network_abr(norm_Connection, pat_Connection)

abrConn = edgeslist_to_abr_conn(pat_Connection, HUP_atlasAll)  # Converts edge list to connectivity matrix
percentile_thres = 75
iEEGhupAll_z = node_abr_edge(abrConn, iEEGhupAll_z, percentile_thres)

# probably a dupe but needs double-checking
# import numpy as np
# import nibabel as nib
# import pandas as pd
# import os

# # Import custom modules
# from mergeROIs import merge_rois
# from mainUnivar import main_univar
# from univariateAbr import univariate_abr
# from mainNetwork import main_network
# from networkAbr import network_abr
# from edgeslist2AbrConn import edgeslist_to_abr_conn
# from nodeAbrEdge import node_abr_edge
# from implant2roi import implant2roi
# from make_seizure_free_abr import make_seizure_free_abr
# from make_seizure_free import make_seizure_free

# # Set up paths
# data_path = '/Users/tereza/nishant/atlas/epi_iEEG_atlas/Data'

# # Correct path usage with os.path.join for all files
# aal_nii_path = os.path.join(data_path, 'AAL.nii.gz')
# # roi_aal_mat_path = os.path.join(data_path, 'roiAAL.mat')
# roi_aal_csv_path = os.path.join(data_path, 'roiAAL.csv')
# custom_atlas_xlsx_path = os.path.join(data_path, 'custom_atlas.xlsx')
# mni_atlas_npy_path = os.path.join(data_path, 'MNI_atlas_orig.npy')
# metadata_csv_path = os.path.join(data_path, 'metaData.csv')
# hup_atlas_npy_path = os.path.join(data_path, 'HUP_atlas.npy')
# pat_conn_red_aal_npy_path = os.path.join(data_path, 'pat_ConnectionRedAAL_z.npy')

# # Load a parcellation scheme and metaData
# atlas = {
#     'data': nib.load(aal_nii_path).get_fdata(),
#     'hdr': nib.load(aal_nii_path).header
# }
# atlas['data'][atlas['data'] > 9000] = 0
# # roiAAL = pd.read_csv(roi_aal_mat_path, sep='\t')
# # roiAAL = roiAAL.iloc[:90, :]
# roiAAL = pd.read_csv(roi_aal_csv_path, sep=',')  # Use ',' for CSV
# roiAAL = roiAAL.iloc[:90, :]
# customAAL = pd.read_excel(custom_atlas_xlsx_path)
# atlas['tbl'] = roiAAL
# atlasCustom, roiAALcustom = merge_rois(customAAL, roiAAL, atlas)
# atlasCustom['tbl'] = roiAALcustom
# atlas = atlasCustom

# # Load data
# MNI_atlas = np.load(mni_atlas_npy_path, allow_pickle=True).item()
# metaData = pd.read_csv(metadata_csv_path)
# HUP_atlasAll = np.load(hup_atlas_npy_path, allow_pickle=True).item()
# EngelSFThres = 1.1
# spikeThresh = 24  # empirical, 1 spike/hour

# # Univariate normative modeling of HUP-MNI
# norm_MNI_HUP_Atlas, iEEGhupAll, abrnormHUPAtlas = main_univar(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)
# iEEGhupAll_z = univariate_abr(norm_MNI_HUP_Atlas, iEEGhupAll)

# # Multivariate normative modeling of HUP-MNI
# norm_Connection, norm_ConnectionCount, pat_Connection, pat_ConnectionCount = main_network(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh)

# try:
#     pat_Connection = np.load(pat_conn_red_aal_npy_path, allow_pickle=True).item()
# except FileNotFoundError:
#     print(f"File not found: {pat_conn_red_aal_npy_path}")
#     pat_Connection = network_abr(norm_Connection, pat_Connection)

# abrConn = edgeslist_to_abr_conn(pat_Connection, HUP_atlasAll)  # Converts edge list to connectivity matrix
# percentile_thres = 75
# iEEGhupAll_z = node_abr_edge(abrConn, iEEGhupAll_z, percentile_thres)

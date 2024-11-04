import scipy.io as sc
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, lfilter
from scipy.signal.windows import hamming
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore, ttest_ind
from pathlib import Path
import h5py

#all custom functions 

# Python translation of MATLAB's `mergeROIs` function
def merge_rois(customAAL, roiAAL, atlas):
    
    # Convert inputs into appropriate types (assuming customAAL and roiAAL are Pandas DataFrames)
    for roi in range(len(customAAL)):
        # Assign parcel1 from roiAAL based on customAAL.Roi1
        customAAL.loc[roi, 'parcel1'] = roiAAL.loc[customAAL.loc[roi, 'Roi1'], 'parcel']
        
        # Handle Roi2 assignment
        if np.isnan(customAAL.loc[roi, 'Roi2']):
            customAAL.loc[roi, 'parcel2'] = np.nan
        else:
            customAAL.loc[roi, 'parcel2'] = roiAAL.loc[customAAL.loc[roi, 'Roi2'], 'parcel']
            atlas['data'][atlas['data'] == customAAL.loc[roi, 'parcel2']] = customAAL.loc[roi, 'parcel1']
        
        # Handle Roi3 assignment
        if np.isnan(customAAL.loc[roi, 'Roi3']):
            customAAL.loc[roi, 'parcel3'] = np.nan
        else:
            customAAL.loc[roi, 'parcel3'] = roiAAL.loc[customAAL.loc[roi, 'Roi3'], 'parcel']
            atlas['data'][atlas['data'] == customAAL.loc[roi, 'parcel3']] = customAAL.loc[roi, 'parcel1']
    
    # Handle inclusion/exclusion logic
    included = np.concatenate((customAAL['Roi1'], customAAL['Roi2'], customAAL['Roi3']))
    included = included[~np.isnan(included)]  # Remove NaN values
    
    excluded = np.setxor1d(roiAAL['Sno'], included)
    atlas['data'][np.isin(atlas['data'], roiAAL['parcel'][excluded])] = 0

    # Create atlasCustom and roiAALcustom as outputs
    atlasCustom = atlas
    
    roiAALcustom = {}
    roiAALcustom['Sno'] = np.arange(1, len(customAAL) + 1)
    roiAALcustom['Regions'] = customAAL['Roi_name']
    roiAALcustom['Lobes'] = customAAL['Lobes']
    roiAALcustom['isSideLeft'] = customAAL['Roi_name'].str.endswith('_L')
    roiAALcustom['parcel'] = customAAL['parcel1']
    
    # Calculate coordinates (CRS to RAS transformation)
    xyz = []
    for roi in range(len(customAAL)):
        indices = np.argwhere(atlas['data'] == roiAALcustom['parcel'][roi])
        CRS = np.hstack([indices, np.full((indices.shape[0], 1), roiAALcustom['parcel'][roi])])
        
        RAS = np.dot(atlas['hdr']['Transform']['T'].T, np.hstack([CRS[:, :3], np.ones((CRS.shape[0], 1))]).T).T
        RAS = RAS[:, :3]
        xyz.append(RAS.mean(axis=0))
    xyz = np.array(xyz)
    
    roiAALcustom['x'] = xyz[:, 0]
    roiAALcustom['y'] = xyz[:, 1]
    roiAALcustom['z'] = xyz[:, 2]

    # Convert roiAALcustom to Pandas DataFrame for easier use
    roiAALcustom = pd.DataFrame(roiAALcustom)
    
    return atlasCustom, roiAALcustom

# Python translation of MATLAB's `implant2ROI` function
def implant2roi(atlas, electrodeCord, patientNum):

    # Get unique ROI from atlas excluding zeros
    nROI = np.unique(atlas['data'][atlas['data'] != 0])
    
    # Get voxel coordinates in CRS format
    CRScord = []
    for roi in nROI:
        indices = np.argwhere(atlas['data'] == roi)
        CRS = np.hstack([indices, np.full((indices.shape[0], 1), roi)])
        CRScord.append(CRS)
    CRScord = np.vstack(CRScord)

    # Convert CRS to RAS using atlas transform
    RAScord = np.dot(atlas['hdr']['Transform']['T'].T, np.hstack([CRScord[:, :3], np.ones((CRScord.shape[0], 1))]).T).T
    RAScord = RAScord[:, :3]
    RAScord = np.hstack([RAScord, CRScord[:, 3:]])

    # Identify nearest neighbor of each electrode
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(RAScord[:, :3])
    distances, indices = nbrs.kneighbors(electrodeCord)
    atlasROI = RAScord[indices.flatten(), 3]

    # Map to ROI numbers
    roiNum = [np.where(atlas['tbl']['parcel'] == roi)[0][0] for roi in atlasROI]
    
    # Create electrode2roi table
    electrode2roi = pd.DataFrame({'roiNum': roiNum, 'atlasROI': atlasROI, 'patientNum': patientNum})
    
    return electrode2roi

# Python translation of MATLAB's `getNormPSD` function
def get_norm_psd(iEEGnormal, data_timeS, SamplingFrequency):

    # Get sampling frequency, time domain data, window length, and NFFT
    Fs = SamplingFrequency
    data_seg = data_timeS[:Fs*60, :]
    window = Fs * 2
    NFFT = window

    # Compute PSD
    f, psd = welch(data_seg, fs=Fs, window=hamming(window), nperseg=window, noverlap=0, nfft=NFFT, axis=0)

    # Filter out noise frequency between 57.7Hz and 62.5Hz
    idx = (f >= 57.5) & (f <= 62.5)
    psd[idx, :] = 0
    f = f[~idx]

    # Compute bandpower
    delta = np.trapz(psd[(f >= 1) & (f < 4)], f[(f >= 1) & (f < 4)], axis=0)
    theta = np.trapz(psd[(f >= 4) & (f < 8)], f[(f >= 4) & (f < 8)], axis=0)
    alpha = np.trapz(psd[(f >= 8) & (f < 13)], f[(f >= 8) & (f < 13)], axis=0)
    beta = np.trapz(psd[(f >= 13) & (f < 30)], f[(f >= 13) & (f < 30)], axis=0)
    gamma = np.trapz(psd[(f >= 30) & (f < 80)], f[(f >= 30) & (f < 80)], axis=0)
    broad = np.trapz(psd[(f >= 1) & (f < 80)], f[(f >= 1) & (f < 80)], axis=0)

    # Log transform
    deltalog = np.log10(delta + 1)
    thetalog = np.log10(theta + 1)
    alphalog = np.log10(alpha + 1)
    betalog = np.log10(beta + 1)
    gammalog = np.log10(gamma + 1)
    broadlog = np.log10(broad + 1)

    # Total power
    tPow = deltalog + thetalog + alphalog + betalog + gammalog

    # Relative power
    deltaRel = deltalog / tPow
    thetaRel = thetalog / tPow
    alphaRel = alphalog / tPow
    betaRel = betalog / tPow
    gammaRel = gammalog / tPow

    # Append to iEEGnormal
    iEEGnormal = pd.concat([iEEGnormal, pd.DataFrame({'delta': deltaRel, 'theta': thetaRel, 'alpha': alphaRel, 'beta': betaRel, 'gamma': gammaRel, 'broad': broadlog})], axis=1)

    return iEEGnormal

# Python translation of MATLAB's `getNormEntropy` function
def get_norm_entropy(iEEGnormal, data_timeS, SamplingFrequency):

    # Helper functions for filtering
    def eeg_filter(data, cutoff, fs, btype, order=3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return lfilter(b, a, data, axis=0)

    # Get sampling frequency, time domain data, window length, and NFFT
    Fs = SamplingFrequency
    data_seg = data_timeS[:Fs*60, :]

    # Apply filters
    data_segbb = eeg_filter(data_seg, 80, Fs, 'low')
    data_segbb = eeg_filter(data_segbb, 1, Fs, 'high')
    data_segbb_notch = eeg_filter(data_segbb, 60, Fs, 'bandstop')

    # Compute Shannon entropy
    entropy = np.array([np.log10(-np.sum(p * np.log2(p)) + 1) for p in data_segbb_notch.T])

    # Append entropy to iEEGnormal
    iEEGnormal = pd.concat([iEEGnormal, pd.DataFrame({'entropy': entropy})], axis=1)

    return iEEGnormal


# Python translation of MATLAB's `plotiEEGatlas` function
def plot_ieeg_atlas(iEEGnormal, atlas, plot_option='noplot'):
    
    # Create normAtlas dictionary to store data
    normAtlas = {}
    normAtlas['roi'] = atlas['tbl']['Sno']
    normAtlas['name'] = atlas['tbl']['Regions']
    normAtlas['lobe'] = atlas['tbl']['Lobes']
    normAtlas['isSideLeft'] = atlas['tbl']['isSideLeft']

    # Calculate metrics for each ROI
    nROIs = len(atlas['tbl']['Sno'])
    for roi in range(nROIs):
        idx = iEEGnormal['roiNum'] == roi
        
        # Number of electrodes in each region
        normAtlas.setdefault('nElecs', []).append(np.sum(idx))
        
        # Mean and standard deviation for delta, theta, alpha, beta, gamma, and broad bands
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']:
            normAtlas.setdefault(band, []).append(iEEGnormal[band][idx].tolist())
            normAtlas.setdefault(f'{band}Mean', []).append(np.mean(iEEGnormal[band][idx]))
            normAtlas.setdefault(f'{band}Std', []).append(np.std(iEEGnormal[band][idx]))
    
    # Convert to DataFrame for easier processing
    normAtlas = pd.DataFrame(normAtlas)
    
    # Remove regions with no electrodes
    normAtlas = normAtlas[normAtlas['nElecs'] > 0]

    # Plotting logic
    if plot_option == 'plot':
                
        bands = ['deltaMean', 'thetaMean', 'alphaMean', 'betaMean', 'gammaMean', 'broadMean']
        for i, band in enumerate(bands):
            normAtlas['scaled'] = minmax_scale(normAtlas[band])
            nodeVal = pd.DataFrame({
                'x': normAtlas['x'],
                'y': normAtlas['y'],
                'z': normAtlas['z'],
                'value': normAtlas['scaled'],
                'color': normAtlas['scaled']
            })
            nodeVal.to_csv('nodeVal.node', sep='\t', index=False, header=False)
            
            # Brain network visualization (placeholder for actual implementation)
            if i < 5:
                # Call some equivalent of BrainNet_MapCfg function for plotting
                pass  # Replace with Python visualization library
            else:
                # Call some equivalent of BrainNet_MapCfg function for plotting (high frequency config)
                pass  # Replace with Python visualization library
    
    return normAtlas

# Python translation of MATLAB's `compareiEEGatlas` function
def compare_ieeg_atlas(normMNIAtlas, normHUPAtlas, plot_option='noplot'):
    
    # Filter out matching ROIs
    MNI = normMNIAtlas[normMNIAtlas['roi'].isin(normHUPAtlas['roi'])]
    HUP = normHUPAtlas[normHUPAtlas['roi'].isin(normMNIAtlas['roi'])]
    
    if plot_option == 'plot':
        # Compare the number of electrodes between atlases
        plt.figure()
        plt.barh(np.arange(len(MNI)), [MNI['nElecs'], HUP['nElecs']], label=['MNI', 'HUP'])
        plt.xlabel('Number of electrodes')
        plt.yticks(np.arange(len(MNI)), MNI['name'])
        plt.gca().tick_params(axis='y', which='both', labelsize='small')
        plt.tight_layout()
        plt.savefig('Figure/nElec.pdf', format='pdf', dpi=300)

        # Get the effect size across regions
        plt.figure()
        bands = ['deltaMean', 'thetaMean', 'alphaMean', 'betaMean', 'gammaMean', 'broadMean']
        data = []
        for band in bands:
            data.append(np.vstack([MNI[band].values, HUP[band].values]).T)
        data = np.concatenate(data, axis=1)
        
        plt.boxplot(data[:, :10])  # Simplified scatter representation
        plt.xticks(np.arange(1, 11), [f'{band}MNI' for band in bands[:5]] + [f'{band}HUP' for band in bands[:5]], rotation=45)
        plt.ylabel('Normalized relative bandpower')
        plt.tight_layout()
        plt.savefig('Figure/bandPow.pdf', format='pdf', dpi=300)

    # Combine HUP with MNI for missing ROIs
    newROIhup = normHUPAtlas[~normHUPAtlas['roi'].isin(normMNIAtlas['roi'])]
    normMNIAtlas = pd.concat([normMNIAtlas, newROIhup], ignore_index=True)
    
    # Update metrics for common ROIs
    commonROIhup = normHUPAtlas[normHUPAtlas['roi'].isin(normMNIAtlas['roi'])]
    lbl = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']
    for _, row in commonROIhup.iterrows():
        id = normMNIAtlas.index[normMNIAtlas['roi'] == row['roi']][0]
        normMNIAtlas.at[id, 'nElecs'] += row['nElecs']
        for band in lbl:
            combined = normMNIAtlas.at[id, band] + row[band]
            normMNIAtlas.at[id, band] = combined
            normMNIAtlas.at[id, f'{band}Mean'] = np.mean(combined)
            normMNIAtlas.at[id, f'{band}Std'] = np.std(combined)
    
    normMNIAtlas = normMNIAtlas.sort_values(by='roi').reset_index(drop=True)
    
    return normMNIAtlas

def node_abr_edge(abr_conn, ieeg_hup_all, percentile_thres):
    fbands = [col for col in abr_conn.columns if col.endswith('_z')]
    ieeg_abr = []
    
    for s in range(len(abr_conn)):
        node_abr = []
        for f in fbands:
            adj = abr_conn.loc[s, f]
            node_abr.append(np.percentile(adj, percentile_thres, axis=1))
        ieeg_abr.append(np.column_stack(node_abr))
    
    ieeg_abr = np.vstack(ieeg_abr)
    ieeg_hup_all = pd.concat([ieeg_hup_all, pd.DataFrame(ieeg_abr, columns=[f + '_coh' for f in fbands])], axis=1)
    
    return ieeg_hup_all

def edgeslist_to_abr_conn(pat_connection, hup_atlas_all):
    n_sub = pat_connection['patientNum'].unique()
    fbands = [col for col in pat_connection.columns if col.endswith('_z')]
    abr_conn = {'patientNum': []}

    for s in n_sub:
        abr_conn['patientNum'].append(s)
        n_elec = (hup_atlas_all['patient_no'] == s).sum()
        for f in fbands:
            edges = pat_connection.loc[pat_connection['patientNum'] == s, f].values
            adj = np.reshape(edges, (n_elec, n_elec))
            adj[np.isnan(adj)] = 0
            abr_conn.setdefault(f, []).append(np.abs(adj))
    
    return pd.DataFrame(abr_conn)

def univariate_abr(norm_mni_hup_atlas, ieeg_hup_all):
    rel_pow_z = []

    for n_elec in range(len(ieeg_hup_all)):
        roi_num = ieeg_hup_all.loc[n_elec, 'roiNum']
        norm_mu = norm_mni_hup_atlas.loc[roi_num, ['deltaMean', 'thetaMean', 'alphaMean', 'betaMean', 'gammaMean', 'broadMean']].values
        norm_sigma = norm_mni_hup_atlas.loc[roi_num, ['deltaStd', 'thetaStd', 'alphaStd', 'betaStd', 'gammaStd', 'broadStd']].values
        rel_pow = ieeg_hup_all.loc[n_elec, ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']].values
        rel_pow_z.append(np.abs((rel_pow - norm_mu) / norm_sigma))

    rel_pow_z = np.array(rel_pow_z)
    ieeg_hup_all_z = pd.concat([ieeg_hup_all.iloc[:, :3], pd.DataFrame(rel_pow_z, columns=['delta_z', 'theta_z', 'alpha_z', 'beta_z', 'gamma_z', 'broad_z'])], axis=1)
    
    return ieeg_hup_all_z

def make_seizure_free_abr(hup_atlas_all, meta_data, engel_sf_thres, spike_thresh):
    outcomes = np.nanmax(meta_data[['Engel_6_mo', 'Engel_12_mo']].values, axis=1)
    sf_patients = np.where(outcomes <= engel_sf_thres)[0]

    sf_patients_ieeg = hup_atlas_all['patient_no'].isin(sf_patients)
    resected_sf_ieeg = sf_patients_ieeg & hup_atlas_all['resected_ch']
    soz_spared_sf_ieeg = resected_sf_ieeg & hup_atlas_all['soz_ch']
    abnormal_ieeg = soz_spared_sf_ieeg & (hup_atlas_all['spike_24h'] > spike_thresh)

    hup_abr_atlas = hup_atlas_all.loc[abnormal_ieeg].copy()
    hup_abr_atlas['SamplingFrequency'] = hup_atlas_all['SamplingFrequency']
    
    return hup_abr_atlas

def make_seizure_free(hup_atlas_all, meta_data, engel_sf_thres, spike_thresh):
    outcomes = np.nanmax(meta_data[['Engel_6_mo', 'Engel_12_mo']].values, axis=1)
    sf_patients = np.where(outcomes <= engel_sf_thres)[0]

    sf_patients_ieeg = hup_atlas_all['patient_no'].isin(sf_patients)
    spared_sf_ieeg = sf_patients_ieeg & ~hup_atlas_all['resected_ch']
    not_soz_spared_sf_ieeg = spared_sf_ieeg & ~hup_atlas_all['soz_ch']
    healthy_ieeg = not_soz_spared_sf_ieeg & (hup_atlas_all['spike_24h'] < spike_thresh)

    hup_atlas = hup_atlas_all.loc[healthy_ieeg].copy()
    hup_atlas['SamplingFrequency'] = hup_atlas_all['SamplingFrequency']
    
    return hup_atlas

# Python translation of MATLAB's `mainUnivar` function
def main_univar(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh):
    
    # MNI atlas electrode to ROI
    electrodeCord = MNI_atlas['ChannelPosition']
    patientNum = MNI_atlas['Patient']
    iEEG_mni = implant2roi(atlas, electrodeCord, patientNum)

    # MNI atlas normalised bandpower
    data_MNI = MNI_atlas['Data_W']
    SamplingFrequency = MNI_atlas['SamplingFrequency']
    iEEG_mni = get_norm_psd(iEEG_mni, data_MNI, SamplingFrequency)

    # Seizure-free HUP atlas electrode to ROI
    HUP_atlas = make_seizure_free(HUP_atlasAll, metaData, EngelSFThres, spikeThresh)
    electrodeCord = HUP_atlas['mni_coords']
    patientNum = HUP_atlas['patient_no']
    iEEG_hup = implant2roi(atlas, electrodeCord, patientNum)

    # HUP atlas normalised bandpower
    data_HUP = HUP_atlas['wake_clip']
    SamplingFrequency = HUP_atlas['SamplingFrequency']
    iEEG_hup = get_norm_entropy(iEEG_hup, data_HUP, SamplingFrequency)

    # Visualise MNI and HUP atlas
    norm_mni_atlas = plot_ieeg_atlas(iEEG_mni, atlas, plot_option='noplot')
    norm_hup_atlas = plot_ieeg_atlas(iEEG_hup, atlas, plot_option='noplot')
    norm_MNI_HUP_Atlas = compare_ieeg_atlas(norm_mni_atlas, norm_hup_atlas, plot_option='plot')

    # Seizure-free HUP atlas electrode to ROI for all patients
    electrodeCord = HUP_atlasAll['mni_coords']
    patientNum = HUP_atlasAll['patient_no']
    iEEG_hup_all = implant2roi(atlas, electrodeCord, patientNum)
    data_HUP_all = HUP_atlasAll['wake_clip']
    SamplingFrequency = HUP_atlasAll['SamplingFrequency']
    iEEG_hup_all = get_norm_psd(iEEG_hup_all, data_HUP_all, SamplingFrequency)

    # Abnormal HUP atlas
    HUP_abr_atlas = make_seizure_free_abr(HUP_atlasAll, metaData, EngelSFThres, spikeThresh)
    electrodeCord = HUP_abr_atlas['mni_coords']
    patientNum = HUP_abr_atlas['patient_no']
    iEEG_hup_abr = implant2roi(atlas, electrodeCord, patientNum)
    data_HUP_abr = HUP_abr_atlas['wake_clip']
    SamplingFrequency = HUP_abr_atlas['SamplingFrequency']
    iEEG_hup_abr = get_norm_psd(iEEG_hup_abr, data_HUP_abr, SamplingFrequency)
    abrnorm_hup_atlas = plot_ieeg_atlas(iEEG_hup_abr, atlas, plot_option='noplot')
    abrnorm_hup_atlas = abrnorm_hup_atlas.sort_values(by='nElecs', ascending=False)

    # Address reviewer 1 comment
    # pGrp, d = rev1_actual_pow(HUP_abr_atlas, iEEG_hup_abr, HUP_atlas, MNI_atlas, iEEG_hup, iEEG_mni)
    # d = rev1_surg_outcome(HUP_atlasAll, iEEG_hup_all, metaData)

    return norm_MNI_HUP_Atlas, iEEG_hup_all, abrnorm_hup_atlas
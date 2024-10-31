import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming
from sklearn.neighbors import NearestNeighbors

# Assume the following custom functions are in respective Python files:
from implant2roi import implant2roi
from make_seizure_free import make_seizure_free, make_seizure_free_abr
from norm_psd import get_norm_psd
from norm_entropy import get_norm_entropy
from plot_ieeg_atlas import plot_ieeg_atlas
from compare_ieeg_atlas import compare_ieeg_atlas

def main_univar(metaData, atlas, MNI_atlas, HUP_atlasAll, EngelSFThres, spikeThresh):
    # MNI atlas electrode to ROI
    electrodeCord = MNI_atlas['ChannelPosition']
    patientNum = MNI_atlas['Patient']
    iEEGmni = implant2roi(atlas, electrodeCord, patientNum)

    # MNI atlas normalised bandpower
    data_MNI = MNI_atlas['Data_W']
    SamplingFrequency = MNI_atlas['SamplingFrequency']
    iEEGmni = get_norm_psd(iEEGmni, data_MNI, SamplingFrequency)

    # Seizure free HUP atlas electrode to ROI
    HUP_atlas = make_seizure_free(HUP_atlasAll, metaData, EngelSFThres, spikeThresh)
    electrodeCord = HUP_atlas['mni_coords']
    patientNum = HUP_atlas['patient_no']
    iEEGhup = implant2roi(atlas, electrodeCord, patientNum)

    # HUP atlas normalised bandpower
    data_HUP = HUP_atlas['wake_clip']
    SamplingFrequency = HUP_atlas['SamplingFrequency']
    iEEGhup = get_norm_entropy(iEEGhup, data_HUP, SamplingFrequency)

    # Visualise MNI and HUP atlas
    normMNIAtlas = plot_ieeg_atlas(iEEGmni, atlas, 'noplot')
    normHUPAtlas = plot_ieeg_atlas(iEEGhup, atlas, 'noplot')
    norm_MNI_HUP_Atlas = compare_ieeg_atlas(normMNIAtlas, normHUPAtlas, 'plot')

    # Process all HUP data
    electrodeCord = HUP_atlasAll['mni_coords']
    patientNum = HUP_atlasAll['patient_no']
    iEEGhupAll = implant2roi(atlas, electrodeCord, patientNum)
    data_HUPAll = HUP_atlasAll['wake_clip']
    iEEGhupAll = get_norm_psd(iEEGhupAll, data_HUPAll, SamplingFrequency)

    # Abnormal HUP atlas
    HUP_Abr_atlas = make_seizure_free_abr(HUP_atlasAll, metaData, EngelSFThres, spikeThresh)
    electrodeCord = HUP_Abr_atlas['mni_coords']
    patientNum = HUP_Abr_atlas['patient_no']
    iEEGhupAbr = implant2roi(atlas, electrodeCord, patientNum)
    data_HUPAbr = HUP_Abr_atlas['wake_clip']
    SamplingFrequency = HUP_Abr_atlas['SamplingFrequency']
    iEEGhupAbr = get_norm_psd(iEEGhupAbr, data_HUPAbr, SamplingFrequency)
    abrnormHUPAtlas = plot_ieeg_atlas(iEEGhupAbr, atlas, 'noplot')
    abrnormHUPAtlas = abrnormHUPAtlas.sort_values(by='nElecs', ascending=False)

    # Address reviewer 1 comment
    # pGrp, d = rev1_actual_pow(HUP_Abr_atlas, iEEGhupAbr, HUP_atlas, MNI_atlas, iEEGhup, iEEGmni)
    # d = rev1_surg_outcome(HUP_atlasAll, iEEGhupAll, metaData)

    return norm_MNI_HUP_Atlas, iEEGhupAll, abrnormHUPAtlas


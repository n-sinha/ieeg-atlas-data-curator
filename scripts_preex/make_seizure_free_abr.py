import pandas as pd
import numpy as np

def make_seizure_free_abr(hup_atlas_all, meta_data, engel_sf_thres, spike_thresh):
    """
    Selects abnormal iEEG data from patients with seizure freedom below a certain threshold and with high spike rates.
    """
    outcomes = np.nanmax(meta_data[['Engel_6_mo', 'Engel_12_mo']].values, axis=1)
    sf_patients = outcomes <= engel_sf_thres

    # Index of resected electrodes in seizure-free patients that are also SOZ
    sf_patients_ieeg = hup_atlas_all['patient_no'].isin(meta_data[sf_patients].index)
    resected_sf_ieeg = sf_patients_ieeg & hup_atlas_all['resected_ch']
    soz_spared_sf_ieeg = resected_sf_ieeg & hup_atlas_all['soz_ch']
    abnormal_ieeg = soz_spared_sf_ieeg & (hup_atlas_all['spike_24h'] > spike_thresh)

    # Create subset atlas
    hup_abr_atlas = hup_atlas_all[abnormal_ieeg].copy()
    return hup_abr_atlas
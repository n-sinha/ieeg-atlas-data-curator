import numpy as np

def make_seizure_free(hup_atlas_all, meta_data, engel_sf_thres, spike_thresh):
    """
    Selects healthy iEEG data from patients with seizure freedom below a certain threshold and with low spike rates.
    """
    outcomes = np.nanmax(meta_data[['Engel_6_mo', 'Engel_12_mo']].values, axis=1)
    sf_patients = outcomes <= engel_sf_thres

    # Index of spared electrodes in seizure-free patients that are not SOZ
    sf_patients_ieeg = hup_atlas_all['patient_no'].isin(meta_data[sf_patients].index)
    spared_sf_ieeg = sf_patients_ieeg & ~hup_atlas_all['resected_ch']
    not_soz_spared_sf_ieeg = spared_sf_ieeg & ~hup_atlas_all['soz_ch']
    healthy_ieeg = not_soz_spared_sf_ieeg & (hup_atlas_all['spike_24h'] < spike_thresh)

    # Create subset atlas
    hup_atlas = hup_atlas_all[healthy_ieeg].copy()
    return hup_atlas
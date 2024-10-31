def main_network(meta_data, atlas, mni_atlas, hup_atlas_all, engel_sf_thres, spike_thresh):
    """
    Main network analysis function translating from MATLAB to Python.
    """
    # MNI atlas electrode to ROI
    electrode_cord = mni_atlas['ChannelPosition']
    patient_num = mni_atlas['Patient']
    ieeg_mni = implant2roi(atlas, electrode_cord, patient_num)

    # MNI atlas normalized bandpower
    data_mni = mni_atlas['Data_W']
    sampling_frequency = mni_atlas['SamplingFrequency']
    ieeg_mni = get_norm_psd(ieeg_mni, data_mni, sampling_frequency)

    # Seizure free HUP atlas electrode to ROI
    hup_atlas = make_seizure_free(hup_atlas_all, meta_data, engel_sf_thres, spike_thresh)
    electrode_cord = hup_atlas['mni_coords']
    patient_num = hup_atlas['patient_no']
    ieeg_hup = implant2roi(atlas, electrode_cord, patient_num)

    # HUP atlas normalized bandpower
    data_hup = hup_atlas['wake_clip']
    sampling_frequency = hup_atlas['SamplingFrequency']
    ieeg_hup = get_norm_psd(ieeg_hup, data_hup, sampling_frequency)

    # Make an edge list of normal edges
    cord_hup = hup_atlas['mni_coords']
    cord_mni = mni_atlas['ChannelPosition']

    try:
        # Using pandas to read .mat file
        norm_connection = pd.read_pickle('nom_ConnectionRedAAL.mat')
    except:
        norm_connection = make_edge_list(ieeg_hup, data_hup, cord_hup,
                                       ieeg_mni, data_mni, cord_mni, sampling_frequency)

    # Visualization for normative iEEG
    norm_connection_count = check_sparsity(norm_connection, atlas)

    # Analysis of all HUP patients
    electrode_cord = hup_atlas_all['mni_coords']
    patient_num = hup_atlas_all['patient_no']
    ieeg_hup_all = implant2roi(atlas, electrode_cord, patient_num)

    data_hup_all = hup_atlas_all['wake_clip']
    sampling_frequency = hup_atlas_all['SamplingFrequency']
    ieeg_hup_all = get_norm_psd(ieeg_hup_all, data_hup_all, sampling_frequency)
    cord_hup_all = hup_atlas_all['mni_coords']

    try:
        pat_connection = pd.read_pickle('pat_ConnectionRedAAL.mat')
    except:
        pat_connection = make_edge_list_pat(ieeg_hup_all, data_hup_all, cord_hup_all, sampling_frequency)

    # Visualization of example patient
    pat_connection_count = check_sparsity(pat_connection, atlas)

    return norm_connection, norm_connection_count, pat_connection, pat_connection_count
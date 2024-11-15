import pandas as pd

def filter_seizure_free_patients(ieeg_data, meta_data, seizure_free_threshold=1.1):
    """
    Filter iEEG data to include only seizure-free patients.

    Args:
        ieeg_data (DataFrame): iEEG data with patient numbers.
        meta_data (DataFrame): Patient metadata including seizure outcomes.
        seizure_free_threshold (float): Threshold for considering a patient seizure-free.

    Returns:
        DataFrame: Filtered iEEG data including only seizure-free patients.
    """
    # Merge ieeg_data with meta_data on patient number
    merged_data = ieeg_data.merge(meta_data, left_on='patientNum', right_on='PatientID')

    # Filter based on seizure outcome (assuming 'Engel_6_mo' is your outcome variable)
    seizure_free = merged_data['Engel_6_mo'] <= seizure_free_threshold
    filtered_data = merged_data[seizure_free]

    # Return only the ieeg_data columns
    return filtered_data[ieeg_data.columns]

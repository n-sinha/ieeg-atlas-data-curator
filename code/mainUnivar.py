def univariate_abr(norm_mni_hup_atlas, ieeg_hup_all):
    """
    Calculate standardized (z-score) power values for different frequency bands.
    
    Parameters:
    norm_mni_hup_atlas: pandas DataFrame with mean and std values for each frequency band
    ieeg_hup_all: pandas DataFrame with raw power values and ROI numbers
    
    Returns:
    ieeg_hup_all_z: pandas DataFrame with standardized power values
    """
    import numpy as np
    import pandas as pd

    # Initialize array for z-scores
    n_elec = len(ieeg_hup_all)
    rel_pow_z = np.zeros((n_elec, 6))  # 6 frequency bands

    # Calculate z-scores for each electrode
    for n_elec in range(len(ieeg_hup_all)):
        # Get ROI number for current electrode
        roi_num = ieeg_hup_all.loc[n_elec, 'roiNum']
        
        # Get normative means for all bands
        norm_mu = np.array([
            norm_mni_hup_atlas.loc[roi_num, 'deltaMean'],
            norm_mni_hup_atlas.loc[roi_num, 'thetaMean'],
            norm_mni_hup_atlas.loc[roi_num, 'alphaMean'],
            norm_mni_hup_atlas.loc[roi_num, 'betaMean'],
            norm_mni_hup_atlas.loc[roi_num, 'gammaMean'],
            norm_mni_hup_atlas.loc[roi_num, 'broadMean']
        ])

        # Get normative standard deviations for all bands
        norm_sigma = np.array([
            norm_mni_hup_atlas.loc[roi_num, 'deltaStd'],
            norm_mni_hup_atlas.loc[roi_num, 'thetaStd'],
            norm_mni_hup_atlas.loc[roi_num, 'alphaStd'],
            norm_mni_hup_atlas.loc[roi_num, 'betaStd'],
            norm_mni_hup_atlas.loc[roi_num, 'gammaStd'],
            norm_mni_hup_atlas.loc[roi_num, 'broadStd']
        ])

        # Get relative power values for current electrode
        rel_pow = np.array([
            ieeg_hup_all.loc[n_elec, 'delta'],
            ieeg_hup_all.loc[n_elec, 'theta'],
            ieeg_hup_all.loc[n_elec, 'alpha'],
            ieeg_hup_all.loc[n_elec, 'beta'],
            ieeg_hup_all.loc[n_elec, 'gamma'],
            ieeg_hup_all.loc[n_elec, 'broad']
        ])

        # Calculate z-scores
        rel_pow_z[n_elec, :] = (rel_pow - norm_mu) / norm_sigma

    # Take absolute value of abnormality
    rel_pow_z = np.abs(rel_pow_z)

    # Create output DataFrame
    # First get the first 3 columns of original DataFrame
    ieeg_hup_all_z = ieeg_hup_all.iloc[:, :3].copy()
    
    # Add z-score columns
    z_scores_df = pd.DataFrame(
        rel_pow_z,
        columns=['delta_z', 'theta_z', 'alpha_z', 'beta_z', 'gamma_z', 'broad_z']
    )
    
    # Concatenate original columns with z-scores
    ieeg_hup_all_z = pd.concat([ieeg_hup_all_z, z_scores_df], axis=1)

    return ieeg_hup_all_z
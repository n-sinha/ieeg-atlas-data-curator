import pandas as pd
import statsmodels.formula.api as smf

def mixed_effects_model_group(ieeg_z_data, spectral_feature, group_variable='Group'):
    """
    Fit a mixed-effects model for a given spectral feature z-score, including group comparisons.

    Args:
        ieeg_z_data (DataFrame): DataFrame with z-scores per electrode, including 'patientNum', 'roiNum', and group variable.
        spectral_feature (str): The spectral feature to analyze (e.g., 'deltaRel').
        group_variable (str): The column name representing the group variable.

    Returns:
        MixedLMResults: The fitted model results.
    """
    z_col = f"{spectral_feature}_z"
    ieeg_z_data['patientNum'] = ieeg_z_data['patientNum'].astype('category')
    ieeg_z_data['roiNum'] = ieeg_z_data['roiNum'].astype('category')
    ieeg_z_data[group_variable] = ieeg_z_data[group_variable].astype('category')

    # Mixed-effects model with Group as fixed effect and random intercepts for patient and ROI
    model = smf.mixedlm(f"{z_col} ~ {group_variable}", ieeg_z_data, groups=ieeg_z_data["patientNum"], re_formula="~roiNum")
    result = model.fit()
    return result
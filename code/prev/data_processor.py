import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt

class iEEGProcessor:
    def __init__(self):
        """Initialize iEEG processor with EEG frequency bands."""
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80),
            'broad': (1, 80)
        }

    def _bandpower(self, psd, freqs, freq_range):
        """Calculate power in the given frequency range using the PSD."""
        idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
        band_power = np.trapz(psd[idx], freqs[idx])
        return band_power

    def get_norm_psd(self, iEEGnormal, data_timeS, sampling_frequency=200):
        """
        Compute normalized power spectral densities for different EEG frequency bands.
        
        Args:
            iEEGnormal (DataFrame): DataFrame to append results to.
            data_timeS (array): Time domain EEG data for a single electrode.
            sampling_frequency (int): Sampling frequency of the EEG data.
        
        Returns:
            DataFrame: Updated DataFrame with new EEG features.
        """
        # Define window and NFFT parameters
        window = sampling_frequency * 2
        NFFT = window
        
        # Compute PSD using Welch's method
        f, data_psd = welch(
            data_timeS,
            fs=sampling_frequency, 
            window=hamming(window),
            nfft=NFFT, scaling='density', 
            noverlap=window//2
        )
        
        # Filter out noise frequency (e.g., 60 Hz)
        noise_mask = (f >= 57.5) & (f <= 62.5)
        f = f[~noise_mask]
        data_psd = data_psd[~noise_mask]
        
        # Calculate band powers for each EEG band
        band_powers = {
            band: self._bandpower(data_psd, f, freq_range)
            for band, freq_range in self.bands.items()
        }
        
        # Compute log transform of band powers
        log_band_powers = {
            f'{band}log': np.log10(power + 1)
            for band, power in band_powers.items()
        }
        
        # Calculate total power across all bands
        total_band_power = np.sum([value for value in log_band_powers.values()])
        
        # Calculate relative powers for each band
        relative_band_powers = {
            f'{band}Rel': log_band_powers[f'{band}log'] / total_band_power
            for band in self.bands
        }
        
        # Create DataFrame row with computed features
        data_to_append = pd.DataFrame([relative_band_powers])
        data_to_append['broadlog'] = log_band_powers['broadlog']
        iEEGnormal = pd.concat([iEEGnormal, data_to_append], ignore_index=True)
        
        return iEEGnormal

    def aggregate_features_per_roi(self, iEEG_data, spectral_features):
        """
        Aggregate spectral features per ROI by computing mean and std.
        
        Args:
            iEEG_data (DataFrame): DataFrame with 'roiNum' and spectral features.
            spectral_features (list): List of spectral feature column names.
        
        Returns:
            DataFrame: Aggregated data per ROI with mean and std of spectral features.
        """
        # Group data by ROI and compute mean and std
        aggregated_data = iEEG_data.groupby('roiNum')[spectral_features].agg(['mean', 'std']).reset_index()
        # Flatten MultiIndex columns
        aggregated_data.columns = ['roiNum'] + [f"{feat}_{stat}" for feat in spectral_features for stat in ['mean', 'std']]
        return aggregated_data

    def compute_z_scores(self, iEEG_patient, norm_aggregated, spectral_features):
        """
        Compute z-scores for each electrode in patient data using normative data.
        
        Args:
            iEEG_patient (DataFrame): Patient data with 'roiNum' and spectral features.
            norm_aggregated (DataFrame): Normative data per ROI with mean and std.
            spectral_features (list): List of spectral feature column names.
        
        Returns:
            DataFrame: Patient data with computed z-scores for each spectral feature.
        """
        # Merge patient data with normative data based on ROI
        merged_data = pd.merge(iEEG_patient, norm_aggregated, on='roiNum', how='left')
        for feat in spectral_features:
            mean_col = f"{feat}_mean"
            std_col = f"{feat}_std"
            z_col = f"{feat}_z"
            # Compute z-score
            merged_data[z_col] = (merged_data[feat] - merged_data[mean_col]) / merged_data[std_col]
            # Handle division by zero or NaNs
            merged_data[z_col] = merged_data[z_col].replace([np.inf, -np.inf], np.nan)
            # Take absolute value if desired
            merged_data[z_col] = merged_data[z_col].abs()
        return merged_data

    def aggregate_z_scores_per_roi(self, iEEG_z_data, spectral_features):
        """
        Aggregate z-scores per ROI by averaging over electrodes.
        
        Args:
            iEEG_z_data (DataFrame): DataFrame with z-scores per electrode.
            spectral_features (list): List of spectral feature column names.
        
        Returns:
            DataFrame: Aggregated z-scores per ROI.
        """
        z_score_cols = [f"{feat}_z" for feat in spectral_features]
        aggregated_z = iEEG_z_data.groupby('roiNum')[z_score_cols].mean().reset_index()
        return aggregated_z

    # def plot_z_scores(self, aggregated_z_scores, spectral_feature):
    #     """
    #     Plot mean z-scores per ROI for a given spectral feature.
        
    #     Args:
    #         aggregated_z_scores (DataFrame): Aggregated z-scores per ROI.
    #         spectral_feature (str): Spectral feature to plot.
    #     """
    #     z_col = f"{spectral_feature}_z"
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(aggregated_z_scores['roiNum'], aggregated_z_scores[z_col])
    #     plt.xlabel('ROI Number')
    #     plt.ylabel('Mean Z-score')
    #     plt.title(f'Mean Z-scores for {spectral_feature} Band')
    #     plt.tight_layout()
    #     # plt.savefig(f'z_scores_{spectral_feature}.png')
    #     plt.show()

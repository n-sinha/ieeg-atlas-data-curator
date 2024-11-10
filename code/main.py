from data_processor import iEEGProcessor
from data_loader import AtlasDataLoader
from atlas_mapper import AtlasMapper
from plot_electrode_mappings import plot_electrode_mappings
from plot_electrodes_with_rois import plot_electrodes_with_rois

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Initialize data loader and load data
    base_path = '/Users/tereza/nishant/atlas/epi_iEEG_atlas/Data'
    print("Loading data...")
    loader = AtlasDataLoader(base_path)
    loader.load_atlas_data()
    print("Data loaded successfully.")
    
    # Initialize processor and atlas mapper
    print("Initializing processor and atlas mapper...")
    processor = iEEGProcessor()
    atlas_mapper = AtlasMapper(loader.atlas_data)
    print("Processor and atlas mapper initialized.")
    
    # Map electrodes to ROIs
    print("Mapping electrodes to ROIs for MNI data...")
    mni_electrode_to_roi = atlas_mapper.map_electrodes_to_rois(loader.mni_coords.values, loader.mni_idx_map_arr)
    print("Mapping electrodes to ROIs for HUP data...")
    hup_electrode_to_roi = atlas_mapper.map_electrodes_to_rois(loader.hup_coords.values, loader.hup_idx_map_arr)
    
    # Print the number of electrodes and unique ROIs
    print(f"Number of electrodes in MNI data: {len(mni_electrode_to_roi)}")
    print(f"Number of unique ROIs in MNI data: {mni_electrode_to_roi['roiNum'].nunique()}")
    print(f"Number of electrodes in HUP data: {len(hup_electrode_to_roi)}")
    print(f"Number of unique ROIs in HUP data: {hup_electrode_to_roi['roiNum'].nunique()}")
    
    # Process HUP data
    print("Processing HUP data...")
    hup_iEEGnormal = pd.DataFrame()
    for idx in range(loader.hup_ts.shape[1]):
        hup_electrode_data = loader.hup_ts.iloc[:, idx].values
        hup_iEEGnormal = processor.get_norm_psd(hup_iEEGnormal, hup_electrode_data, sampling_frequency=loader.hup_samp_freq)
        if idx % 500 == 0:
            print(f"Processed {idx} electrodes out of {loader.hup_ts.shape[1]} in HUP data...")
    print("HUP data processing completed.")
    
    # Process MNI data
    print("Processing MNI data...")
    mni_iEEGnormal = pd.DataFrame()
    for idx in range(loader.mni_ts.shape[1]):
        mni_electrode_data = loader.mni_ts.iloc[:, idx].values
        mni_iEEGnormal = processor.get_norm_psd(mni_iEEGnormal, mni_electrode_data, sampling_frequency=loader.mni_samp_freq)
        if idx % 500 == 0:
            print(f"Processed {idx} electrodes out of {loader.mni_ts.shape[1]} in MNI data...")
    print("MNI data processing completed.")
    
    # Combine spectral features with ROI mapping
    spectral_features = ['deltaRel', 'thetaRel', 'alphaRel', 'betaRel', 'gammaRel', 'broadRel', 'broadlog']
    mni_data = pd.concat([mni_electrode_to_roi.reset_index(drop=True), mni_iEEGnormal.reset_index(drop=True)], axis=1)
    hup_data = pd.concat([hup_electrode_to_roi.reset_index(drop=True), hup_iEEGnormal.reset_index(drop=True)], axis=1)
    
    print("First few mappings for MNI data:")
    print(mni_electrode_to_roi.head())
    print("First few mappings for HUP data:")
    print(hup_electrode_to_roi.head())

    # Verify data alignment
    print("Verifying data alignment...")
    if mni_data.isnull().values.any():
        print("Warning: NaN values found in MNI data after concatenation.")
    if hup_data.isnull().values.any():
        print("Warning: NaN values found in HUP data after concatenation.")
    
    # Print examples of features before normalization
    print("First few rows of HUP data before z-score normalization:")
    print(hup_data[spectral_features].head())
    
    # Aggregate normative data per ROI
    print("Aggregating normative data per ROI...")
    norm_mni_aggregated = processor.aggregate_features_per_roi(mni_data, spectral_features)
    print("Normative data aggregation completed.")
    
    # Compute z-scores for HUP data
    print("Computing z-scores for HUP data...")
    hup_data_z = processor.compute_z_scores(hup_data, norm_mni_aggregated, spectral_features)
    print("Z-score computation completed.")
    
    # Print examples of features after normalization
    z_score_cols = [f"{feat}_z" for feat in spectral_features]
    print("First few rows of HUP data after z-score normalization:")
    print(hup_data_z[z_score_cols].head())
    
    # Optional: Aggregate z-scores per ROI
    print("Aggregating z-scores per ROI...")
    aggregated_z_scores = processor.aggregate_z_scores_per_roi(hup_data_z, spectral_features)
    print("Z-score aggregation completed.")
    
    # Print the number of ROIs after aggregation
    print(f"Number of ROIs after aggregation: {len(aggregated_z_scores)}")
    
    # Print aggregated z-scores
    print("Aggregated Z-scores per ROI:")
    print(aggregated_z_scores.head())

    print("Plotting electrode mappings for HUP data...")
    plot_electrode_mappings(loader.hup_coords.values, hup_electrode_to_roi['roiNum'].values, 'HUP Electrode Mappings')
    print("Plotting electrode mappings for MNI data...")
    plot_electrode_mappings(loader.mni_coords.values, mni_electrode_to_roi['roiNum'].values, 'MNI Electrode Mappings')
    
    common_rois = set(hup_data['roiNum']).intersection(set(mni_data['roiNum']))
    print(f"Common ROIs between HUP and MNI cohorts: {common_rois}")

    # for feat in spectral_features:
    #     processor.plot_z_scores(aggregated_z_scores, feat)
    
    # plt.show() 

    # Plot HUP electrodes
    plot_electrodes_with_rois(loader.hup_coords.values, hup_electrode_to_roi['roiNum'].values, 'HUP')

    # Plot MNI electrodes
    plot_electrodes_with_rois(loader.mni_coords.values, mni_electrode_to_roi['roiNum'].values, 'MNI')

if __name__ == "__main__":
    main()

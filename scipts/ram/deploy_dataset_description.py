#!/usr/bin/env python3
"""
Deploy corrected dataset_description.json to all subjects' ieeg_recon derivatives.

This script:
1. Finds all subjects with ieeg_recon derivatives
2. For each subject, finds the actual electrode file path
3. Creates a subject-specific dataset_description.json with correct paths
4. Deploys it to each subject's ieeg_recon directory
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_electrode_file(subject_dir: Path) -> str:
    """
    Find the actual electrode file path for a subject.
    
    Args:
        subject_dir: Path to the subject's directory
        
    Returns:
        str: BIDS-compliant path to the electrode file
    """
    # Look for electrode files in the ieeg-clips derivative
    ieeg_clips_dir = subject_dir / 'derivatives' / 'ieeg-clips'
    
    if not ieeg_clips_dir.exists():
        logging.warning(f"No ieeg-clips directory found for {subject_dir.name}")
        return None
    
    # Find electrode files in ses-task/ses-0/ieeg/ subdirectories
    electrode_files = list(ieeg_clips_dir.rglob("**/ses-task/ses-0/ieeg/*_electrodes.tsv"))
    
    if not electrode_files:
        logging.warning(f"No electrode files found for {subject_dir.name}")
        return None
    
    # Use the first electrode file found
    electrode_file = electrode_files[0]
    
    # Convert to BIDS-compliant path
    # Extract the relative path from the subject directory
    relative_path = electrode_file.relative_to(subject_dir)
    
    # Convert to BIDS format
    bids_path = f"bids::{relative_path}"
    
    logging.info(f"Found electrode file for {subject_dir.name}: {bids_path}")
    return bids_path

def create_subject_dataset_description(subject_id: str, electrode_file_path: str) -> Dict[str, Any]:
    """
    Create a subject-specific dataset_description.json.
    
    Args:
        subject_id: Subject identifier (e.g., sub-R1010J)
        electrode_file_path: BIDS path to the electrode file
        
    Returns:
        Dict containing the dataset description
    """
    return {
        "Name": "iEEG Reconstruction Derivatives",
        "BIDSVersion": "1.10.0",
        "Description": "Intracranial electrode reconstruction and localization pipeline derivatives including electrode detection, native space localization, ROI mapping, and MNI152 standard space transformation",
        "DatasetType": "derivative",
        "PipelineDescription": {
            "Name": "iEEG-recon",
            "Version": "latest",
            "Description": "Fast and scalable pipeline for accurate reconstruction of intracranial electrodes and implantable devices"
        },
        "SourceDatasets": [
            {
                "URL": electrode_file_path,
                "DOI": "",
                "Description": "Electrode coordinates in MNI152NLin6ASym space from ieeg-clips derivative"
            },
            {
                "URL": "bids::derivatives/ieeg_recon/module2/T1.nii.gz",
                "DOI": "",
                "Description": "fsaverage template space"
            }
        ],
        "GeneratedBy": [
            {
                "Name": "iEEG-recon",
                "Version": "latest",
                "Description": "Fast and scalable pipeline for accurate reconstruction of intracranial electrodes and implantable devices",
                "CodeURL": "https://github.com/nishantsinha89/ieeg_recon",
                "Container": {
                    "Type": "Docker",
                    "Tag": "nishantsinha89/ieeg_recon:latest"
                },
                "Citation": "Lucas A, Scheid BH, Pattnaik AR, Gallagher R, Mojena M, Tranquille A, Prager B, Gleichgerrcht E, Gong R, Litt B, Davis KA, Das S, Stein JM, Sinha N. iEEG-recon: A fast and scalable pipeline for accurate reconstruction of intracranial electrodes and implantable devices. Epilepsia. 2024 Mar;65(3):817-829. doi: 10.1111/epi.17863. Epub 2024 Jan 10. PMID: 38148517; PMCID: PMC10948311."
            }
        ],
        "PipelineSteps": [
            {
                "Name": "Module1_ElectrodeDetection",
                "Description": "Electrode detection and naming from MNI152NLin6ASym coordinates",
                "Inputs": [f"sub-{subject_id}_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv"],
                "Outputs": ["electrode_names.txt"]
            },
            {
                "Name": "Module2_FSAverage_Space_Transformation", 
                "Description": "Transformation of electrodes from MNI152NLin6ASym to fsaverage space using tal.x, tal.y, tal.z coordinates",
                "Inputs": [f"sub-{subject_id}_ses-0_task-FR1_space-MNI152NLin6ASym_electrodes.tsv", "T1.nii.gz (fsaverage template)"],
                "Outputs": ["electrodes_inMRImm.txt", "Quality control images"]
            },
            {
                "Name": "Module3_NativeSpace_ROI_Mapping",
                "Description": "Electrode-to-ROI mapping in fsaverage space using Desikan-Killiany atlas",
                "Inputs": ["electrodes_inMRImm.txt", "aparc+aseg.nii.gz"],
                "Outputs": ["electrodes2ROI.csv", "electrode visualization"]
            },
            {
                "Name": "Module4_MNI152_Transformation",
                "Description": "Transformation of electrodes to MNI152NLin2009cAsym standard space",
                "Inputs": ["electrodes2ROI.csv", "ANTs transformation files"],
                "Outputs": ["electrodes2ROI_mni.csv", "MNI152 registration files"]
            }
        ],
        "Contact Author": [
            "Nishant Sinha <Nishant.Sinha@Pennmedicine.upenn.edu>"
        ],
        "Funding": [
            "NINDS K99NS138680 (PI: Nishant Sinha)"
        ]
    }

def deploy_dataset_description(bids_root: Path):
    """
    Deploy dataset_description.json to all subjects' ieeg_recon derivatives.
    
    Args:
        bids_root: Path to the BIDS root directory
    """
    logging.info(f"Starting deployment of dataset_description.json to {bids_root}")
    
    # Find all subjects with ieeg_recon derivatives
    subjects = []
    for subject_dir in bids_root.glob("sub-*"):
        ieeg_recon_dir = subject_dir / 'derivatives' / 'ieeg_recon'
        if ieeg_recon_dir.exists():
            subjects.append(subject_dir)
    
    logging.info(f"Found {len(subjects)} subjects with ieeg_recon derivatives")
    
    successful_deployments = 0
    failed_deployments = 0
    
    for subject_dir in subjects:
        subject_id = subject_dir.name
        logging.info(f"Processing {subject_id}")
        
        try:
            # Find the electrode file for this subject
            electrode_file_path = find_electrode_file(subject_dir)
            
            if not electrode_file_path:
                logging.warning(f"Skipping {subject_id} - no electrode file found")
                failed_deployments += 1
                continue
            
            # Create subject-specific dataset description
            dataset_description = create_subject_dataset_description(subject_id, electrode_file_path)
            
            # Write to the ieeg_recon directory
            output_file = subject_dir / 'derivatives' / 'ieeg_recon' / 'dataset_description.json'
            
            with open(output_file, 'w') as f:
                json.dump(dataset_description, f, indent=4)
            
            logging.info(f"Successfully deployed dataset_description.json for {subject_id}")
            successful_deployments += 1
            
        except Exception as e:
            logging.error(f"Failed to deploy dataset_description.json for {subject_id}: {e}")
            failed_deployments += 1
    
    logging.info(f"Deployment completed: {successful_deployments} successful, {failed_deployments} failed")

def main():
    """Main function."""
    setup_logging()
    
    # Define BIDS root directory
    bids_root = Path("/users/nishants/ieeg-atlas-data-curator/data/output/ram/BIDS")
    
    if not bids_root.exists():
        logging.error(f"BIDS root directory not found: {bids_root}")
        return
    
    deploy_dataset_description(bids_root)

if __name__ == "__main__":
    main()

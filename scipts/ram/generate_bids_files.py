#!/usr/bin/env python3
"""
BIDS File Generator for Individual Subjects

This script generates BIDS-compliant files for each subject in the output BIDS directory.
For each subject, it creates three files in the primary directory:
1. dataset_description.json - BIDS 1.10 compliant with modifications
2. participants.json - JSON schema for participant metadata
3. participants.tsv - Tab-separated participant data (participant_id, age, sex)

The script matches subjects from the output BIDS directory with participants 
from the input datasets (ds004789, ds004809, ds004865, ds005059, ds005411).

Author: Nishant Sinha
Email: Nishant.Sinha@Pennmedicine.upenn.edu
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BIDSFileGenerator:
    """
    A class to generate BIDS-compliant files for individual subjects.
    
    This class handles:
    - Loading participant data from multiple input datasets
    - Matching subjects with their metadata
    - Generating BIDS-compliant files for each subject
    """
    
    def __init__(self, 
                 bids_output_dir: str = "/users/nishants/ieeg-atlas-data-curator/data/output/ram/BIDS",
                 input_data_dir: str = "/users/nishants/ieeg-atlas-data-curator/data/input/ram"):
        """
        Initialize the BIDS file generator.
        
        Args:
            bids_output_dir: Path to the output BIDS directory
            input_data_dir: Path to the input data directory containing datasets
        """
        self.bids_output_dir = Path(bids_output_dir)
        self.input_data_dir = Path(input_data_dir)
        
        # List of dataset directories to search for participants
        self.dataset_dirs = ["ds004789", "ds004809", "ds004865", "ds005059", "ds005411"]
        
        # Store participant data from all datasets
        self.participant_data = {}
        self.dataset_descriptions = {}
        
        # Load all participant data
        self._load_participant_data()
    
    def _load_participant_data(self) -> None:
        """
        Load participant data from all input datasets.
        
        This method reads participants.tsv and dataset_description.json files
        from each dataset directory and stores them for later use.
        """
        print("Loading participant data from input datasets...")
        
        for dataset in self.dataset_dirs:
            dataset_path = self.input_data_dir / dataset
            
            # Check if dataset directory exists
            if not dataset_path.exists():
                print(f"Warning: Dataset directory {dataset} not found, skipping...")
                continue
            
            # Load participants.tsv
            participants_file = dataset_path / "participants.tsv"
            if participants_file.exists():
                try:
                    df = pd.read_csv(participants_file, sep='\t')
                    # Store participant data with dataset source
                    for _, row in df.iterrows():
                        participant_id = row['participant_id']
                        self.participant_data[participant_id] = {
                            'age': row.get('age', 'n/a'),
                            'sex': row.get('sex', 'n/a'),
                            'hand': row.get('hand', 'n/a'),
                            'dataset_source': dataset
                        }
                    print(f"Loaded {len(df)} participants from {dataset}")
                except Exception as e:
                    print(f"Error loading participants from {dataset}: {e}")
            
            # Load dataset_description.json
            desc_file = dataset_path / "dataset_description.json"
            if desc_file.exists():
                try:
                    with open(desc_file, 'r') as f:
                        self.dataset_descriptions[dataset] = json.load(f)
                    print(f"Loaded dataset description from {dataset}")
                except Exception as e:
                    print(f"Error loading dataset description from {dataset}: {e}")
        
        print(f"Total participants loaded: {len(self.participant_data)}")
    
    def _find_participant_info(self, subject_id: str) -> Optional[Dict]:
        """
        Find participant information for a given subject ID.
        
        Args:
            subject_id: The subject ID to search for (e.g., 'sub-R1010J')
            
        Returns:
            Dictionary containing participant information or None if not found
        """
        return self.participant_data.get(subject_id)
    
    def _create_dataset_description(self, subject_id: str, participant_info: Dict) -> Dict:
        """
        Create a BIDS 1.10 compliant dataset_description.json for a subject.
        
        Args:
            subject_id: The subject ID
            participant_info: Dictionary containing participant information
            
        Returns:
            Dictionary representing the dataset description
        """
        # Get the original dataset description from the source dataset
        source_dataset = participant_info.get('dataset_source', 'unknown')
        original_desc = self.dataset_descriptions.get(source_dataset, {})
        
        # Create modified dataset description
        dataset_desc = {
            "Name": f"Individual Subject Dataset: {subject_id}",
            "BIDSVersion": "1.10.0",  # Updated to BIDS 1.10
            "DatasetType": "raw",
            "Authors": original_desc.get("Authors", ["Unknown Authors"]),  # Keep original authors
            "Funding": original_desc.get("Funding", []) + ["NINDS K99NS138680 (PI: Nishant Sinha)"],  # Preserve original funding and add new
            "License": original_desc.get("License", "CC0"),
            "Acknowledgements": original_desc.get("Acknowledgements", ""),
            "DatasetDOI": original_desc.get("DatasetDOI", ""),
            "GeneratedBy": [
                {
                    "Name": "BIDS Individual Subject Generator",
                    "Version": "1.0.0",
                    "Description": "Script to generate individual subject BIDS files",
                    "Author": "Nishant Sinha <Nishant.Sinha@Pennmedicine.upenn.edu>"
                }
            ],
            "SourceDatasets": [
                {
                    "DOI": original_desc.get("DatasetDOI", ""),
                    "URL": f"https://openneuro.org/datasets/{source_dataset}",
                    "Description": f"Original dataset: {original_desc.get('Name', 'Unknown')}"
                }
            ],
            "ModifiedBy": "Nishant Sinha <Nishant.Sinha@Pennmedicine.upenn.edu>"
        }
        
        return dataset_desc
    
    def _create_participants_json(self) -> Dict:
        """
        Create the participants.json schema file.
        
        Returns:
            Dictionary representing the participants.json schema
        """
        participants_schema = {
            "participant_id": {
                "Description": "Unique participant identifier",
                "LongName": "Participant Identifier"
            },
            "age": {
                "Description": "Age of the participant at time of testing",
                "Units": "years",
                "LongName": "Age"
            },
            "sex": {
                "Description": "Biological sex of the participant",
                "Levels": {
                    "F": "female",
                    "M": "male"
                },
                "LongName": "Sex"
            }
        }
        
        return participants_schema
    
    def _create_participants_tsv(self, subject_id: str, participant_info: Dict) -> pd.DataFrame:
        """
        Create the participants.tsv data for a subject.
        
        Args:
            subject_id: The subject ID
            participant_info: Dictionary containing participant information
            
        Returns:
            DataFrame containing participant data
        """
        # Create DataFrame with required columns
        data = {
            'participant_id': [subject_id],
            'age': [participant_info.get('age', 'n/a')],
            'sex': [participant_info.get('sex', 'n/a')]
        }
        
        return pd.DataFrame(data)
    
    def generate_files_for_subject(self, subject_id: str) -> bool:
        """
        Generate BIDS files for a single subject.
        
        Args:
            subject_id: The subject ID to generate files for
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Generating BIDS files for {subject_id}...")
        
        # Find participant information
        participant_info = self._find_participant_info(subject_id)
        if not participant_info:
            print(f"Warning: No participant information found for {subject_id}")
            return False
        
        # Create primary directory
        primary_dir = self.bids_output_dir / subject_id / "primary"
        primary_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate dataset_description.json
            dataset_desc = self._create_dataset_description(subject_id, participant_info)
            desc_file = primary_dir / "dataset_description.json"
            with open(desc_file, 'w') as f:
                json.dump(dataset_desc, f, indent=4)
            print(f"Created dataset_description.json for {subject_id}")
            
            # Generate participants.json
            participants_json = self._create_participants_json()
            json_file = primary_dir / "participants.json"
            with open(json_file, 'w') as f:
                json.dump(participants_json, f, indent=4)
            print(f"Created participants.json for {subject_id}")
            
            # Generate participants.tsv
            participants_df = self._create_participants_tsv(subject_id, participant_info)
            tsv_file = primary_dir / "participants.tsv"
            participants_df.to_csv(tsv_file, sep='\t', index=False)
            print(f"Created participants.tsv for {subject_id}")
            
            return True
            
        except Exception as e:
            print(f"Error generating files for {subject_id}: {e}")
            return False
    
    def generate_all_files(self) -> None:
        """
        Generate BIDS files for all subjects in the output directory.
        
        This method iterates through all subject directories in the BIDS output
        directory and generates the required files for each subject.
        """
        print("Starting BIDS file generation for all subjects...")
        
        # Get all subject directories
        subject_dirs = [d for d in self.bids_output_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        
        print(f"Found {len(subject_dirs)} subject directories")
        
        successful = 0
        failed = 0
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            
            if self.generate_files_for_subject(subject_id):
                successful += 1
            else:
                failed += 1
        
        print(f"\nBIDS file generation completed!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total subjects processed: {successful + failed}")


def main():
    """
    Main function to run the BIDS file generator.
    
    This function creates an instance of BIDSFileGenerator and runs
    the file generation process for all subjects.
    """
    print("BIDS Individual Subject File Generator")
    print("=" * 50)
    
    # Create generator instance
    generator = BIDSFileGenerator()
    
    # Generate files for all subjects
    generator.generate_all_files()
    
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()

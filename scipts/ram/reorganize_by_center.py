#!/usr/bin/env python3
"""
BIDS Dataset Reorganization by Center

This script reorganizes the BIDS dataset by grouping subjects according to 
the center/hospital where the data was collected, based on the subject name endings.

Center Mapping:
- P → Penn (Hospital of the University of Pennsylvania)
- J → Jefferson (Thomas Jefferson University Hospital)
- D → Dartmouth (Dartmouth-Hitchcock Medical Center)
- M → Mayo (Mayo Clinic)
- N → NINDS (NIH NINDS)
- C → Columbia (Columbia University Hospital)
- E → Emory (Emory University Hospital)
- T → TexasSouthwestern (University of Texas Southwestern Medical Center)

Author: Nishant Sinha
Email: Nishant.Sinha@Pennmedicine.upenn.edu
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class BIDSCenterReorganizer:
    """
    A class to reorganize BIDS dataset by collection center/hospital.
    
    This class handles:
    - Mapping subject name endings to center names
    - Creating center directories
    - Moving subject directories to appropriate centers
    - Preserving all internal file structure
    """
    
    def __init__(self, 
                 bids_dir: str = "/users/nishants/ieeg-atlas-data-curator/data/output/ram/BIDS"):
        """
        Initialize the BIDS center reorganizer.
        
        Args:
            bids_dir: Path to the BIDS directory containing subject folders
        """
        self.bids_dir = Path(bids_dir)
        
        # Center mapping based on subject name endings
        self.center_mapping = {
            'P': 'Penn',
            'J': 'Jefferson', 
            'D': 'Dartmouth',
            'M': 'Mayo',
            'N': 'NINDS',
            'C': 'Columbia',
            'E': 'Emory',
            'T': 'TexasSouthwestern'
        }
        
        # Statistics tracking
        self.stats = {
            'total_subjects': 0,
            'moved_subjects': 0,
            'failed_moves': 0,
            'center_counts': {}
        }
    
    def _get_center_from_subject(self, subject_name: str) -> str:
        """
        Determine the center name from a subject name.
        
        Args:
            subject_name: The subject directory name (e.g., 'sub-R1010J')
            
        Returns:
            The center name or 'Unknown' if not recognized
        """
        # Extract the last character before the directory name
        if subject_name.startswith('sub-'):
            subject_id = subject_name[4:]  # Remove 'sub-' prefix
            last_char = subject_id[-1]  # Get last character
            return self.center_mapping.get(last_char, 'Unknown')
        return 'Unknown'
    
    def _get_subject_directories(self) -> List[Path]:
        """
        Get all subject directories in the BIDS folder.
        
        Returns:
            List of Path objects for subject directories
        """
        subject_dirs = []
        
        if not self.bids_dir.exists():
            print(f"Error: BIDS directory {self.bids_dir} does not exist!")
            return subject_dirs
        
        # Find all directories that start with 'sub-'
        for item in self.bids_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subject_dirs.append(item)
        
        return sorted(subject_dirs)
    
    def _create_center_directory(self, center_name: str) -> Path:
        """
        Create a center directory if it doesn't exist.
        
        Args:
            center_name: Name of the center directory to create
            
        Returns:
            Path to the center directory
        """
        center_dir = self.bids_dir / center_name
        center_dir.mkdir(exist_ok=True)
        return center_dir
    
    def _move_subject_to_center(self, subject_dir: Path, center_name: str) -> bool:
        """
        Move a subject directory to the appropriate center directory.
        
        Args:
            subject_dir: Path to the subject directory
            center_name: Name of the center directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            center_dir = self._create_center_directory(center_name)
            destination = center_dir / subject_dir.name
            
            # Check if destination already exists
            if destination.exists():
                print(f"Warning: {destination} already exists, skipping {subject_dir.name}")
                return False
            
            # Move the directory
            shutil.move(str(subject_dir), str(destination))
            print(f"Moved {subject_dir.name} → {center_name}/")
            return True
            
        except Exception as e:
            print(f"Error moving {subject_dir.name} to {center_name}: {e}")
            return False
    
    def reorganize_dataset(self) -> None:
        """
        Reorganize the entire BIDS dataset by center.
        
        This method:
        1. Scans all subject directories
        2. Determines center for each subject
        3. Creates center directories
        4. Moves subjects to appropriate centers
        5. Provides summary statistics
        """
        print("BIDS Dataset Reorganization by Center")
        print("=" * 50)
        
        # Get all subject directories
        subject_dirs = self._get_subject_directories()
        
        if not subject_dirs:
            print("No subject directories found!")
            return
        
        self.stats['total_subjects'] = len(subject_dirs)
        print(f"Found {len(subject_dirs)} subject directories")
        print()
        
        # Process each subject directory
        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            center_name = self._get_center_from_subject(subject_name)
            
            # Update statistics
            if center_name not in self.stats['center_counts']:
                self.stats['center_counts'][center_name] = 0
            self.stats['center_counts'][center_name] += 1
            
            # Move subject to center
            if center_name == 'Unknown':
                print(f"Warning: Unknown center for {subject_name}, skipping...")
                self.stats['failed_moves'] += 1
            else:
                if self._move_subject_to_center(subject_dir, center_name):
                    self.stats['moved_subjects'] += 1
                else:
                    self.stats['failed_moves'] += 1
        
        # Print summary statistics
        self._print_summary()
    
    def _print_summary(self) -> None:
        """
        Print a summary of the reorganization process.
        """
        print("\n" + "=" * 50)
        print("REORGANIZATION SUMMARY")
        print("=" * 50)
        
        print(f"Total subjects processed: {self.stats['total_subjects']}")
        print(f"Successfully moved: {self.stats['moved_subjects']}")
        print(f"Failed moves: {self.stats['failed_moves']}")
        
        print("\nSubjects per center:")
        for center, count in sorted(self.stats['center_counts'].items()):
            print(f"  {center}: {count} subjects")
        
        print("\nNew directory structure:")
        if self.bids_dir.exists():
            for item in sorted(self.bids_dir.iterdir()):
                if item.is_dir():
                    subject_count = len([d for d in item.iterdir() if d.is_dir() and d.name.startswith('sub-')])
                    print(f"  {item.name}/ ({subject_count} subjects)")
    
    def preview_reorganization(self) -> None:
        """
        Preview what the reorganization would do without actually moving files.
        
        This method shows which subjects would be moved to which centers
        without performing the actual moves.
        """
        print("BIDS Dataset Reorganization Preview")
        print("=" * 50)
        
        # Get all subject directories
        subject_dirs = self._get_subject_directories()
        
        if not subject_dirs:
            print("No subject directories found!")
            return
        
        print(f"Found {len(subject_dirs)} subject directories")
        print()
        
        # Group subjects by center
        center_groups = {}
        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            center_name = self._get_center_from_subject(subject_name)
            
            if center_name not in center_groups:
                center_groups[center_name] = []
            center_groups[center_name].append(subject_name)
        
        # Display preview
        print("Preview of reorganization:")
        for center, subjects in sorted(center_groups.items()):
            print(f"\n{center}/ ({len(subjects)} subjects):")
            for subject in sorted(subjects):
                print(f"  {subject}")
        
        print(f"\nTotal: {len(subject_dirs)} subjects across {len(center_groups)} centers")


def main():
    """
    Main function to run the BIDS center reorganizer.
    
    This function creates an instance of BIDSCenterReorganizer and runs
    the reorganization process.
    """
    # Create reorganizer instance
    reorganizer = BIDSCenterReorganizer()
    
    # Ask user for confirmation
    print("This script will reorganize your BIDS dataset by center.")
    print("Subjects will be moved to center-specific directories.")
    print()
    
    # Show preview first
    reorganizer.preview_reorganization()
    
    print("\n" + "=" * 50)
    print("Proceeding with reorganization...")
    reorganizer.reorganize_dataset()
    print("\nReorganization completed!")


if __name__ == "__main__":
    main()

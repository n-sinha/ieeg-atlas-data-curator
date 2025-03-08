#%%
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
import subprocess
from multiprocessing import Pool
from IPython import embed
#%%
class IEEGClipFinder:
    """
    A class for finding and copying interictal iEEG data files from a BIDS-formatted dataset.

    This class helps manage interictal iEEG recordings by identifying files with the most clips
    and copying them to a destination directory. It's specifically designed to work with
    BIDS (Brain Imaging Data Structure) formatted datasets containing iEEG recordings.

    Attributes:
        bids_path (Path): Path to the root of the BIDS dataset directory.
        project_root (Path): Path to the root directory of the project.

    Example:
        >>> bids_path = Path("/path/to/bids/dataset")
        >>> project_root = Path("/path/to/project")
        >>> finder = IEEGClipFinder(bids_path, project_root)
        >>> finder.copy_file_for_subject("sub-RID0031")

    Notes:
        - The BIDS dataset should contain iEEG recordings in the following structure:
          /bids_path/sub-<subject_id>/derivatives/ieeg-portal-clips/
        - Files are expected to follow the naming convention: *interictal_ieeg*day<number>.h5
        - The class uses rsync for file copying operations
    """

    def __init__(self, bids_path: Path, project_root: Path):
        """
        Initialize the IEEGClipFinder with paths to BIDS dataset and project root.

        Args:
            bids_path (Path): Path to the root of the BIDS dataset directory
            project_root (Path): Path to the root directory of the project
        """
        self.bids_path = bids_path
        self.project_root = project_root

    def find_interictal_file_with_most_clips(self, rid: str) -> tuple[Path, int]:
        """
        Find the interictal iEEG file that contains the most clips for a given subject.
    
        This method searches through all interictal iEEG files for a subject and identifies
        the file containing the most clips. It reads H5 files and counts the number of
        entries in each file.

        Args:
            rid (str): Subject ID (e.g., 'sub-RID0031')
    
        Returns:
            tuple[Path, int]: A tuple containing:
                - Path to the interictal file with the most clips
                - Number of clips in the selected file

        Raises:
            FileNotFoundError: If no interictal iEEG clips are found for the subject
        """
        # Construct path to ieeg clips directory
        ieeg_clips = (self.bids_path / rid / 'derivatives' / 'ieeg-portal-clips')

        # Find all interictal ieeg h5 files
        interictal_ieeg_clips = list(ieeg_clips.rglob('*interictal_ieeg*.h5'))

        if len(interictal_ieeg_clips) == 0:
            raise FileNotFoundError(f"No interictal iEEG clips found for subject {rid}")
        
        # Extract day numbers from filenames
        day_num = [int(x.name.split('_')[-1].split('.')[0].replace('day', '')) for x in interictal_ieeg_clips]
        
        # Pair day numbers with file paths and sort by day
        interictal_ieeg_clips = sorted(zip(day_num, interictal_ieeg_clips), key=lambda x: x[0])
        num_clips = []
        
        # Count number of clips in each file
        for idx, (day_num, interictal_ieeg_clip) in enumerate(interictal_ieeg_clips):
            with h5py.File(interictal_ieeg_clip, 'r') as f:
                num_clips.append(len(list(f.keys())))
        
        # Create DataFrame and sort by number of clips
        df = pd.DataFrame({
            'interictal_ieeg_clips': interictal_ieeg_clips,
            'num_clips': num_clips
        }).sort_values(by='num_clips', ascending=False)
        
        # Return the file path with the most clips
        clip_to_use = df['interictal_ieeg_clips'].iloc[0][1]
        num_clips = df['num_clips'].iloc[0]

        return clip_to_use, num_clips

    def copy_file_for_subject(self, subject_id: str):
        """
        Copy the interictal file with the most clips for a single subject.
        
        This method identifies the file with the most clips using find_interictal_file_with_most_clips
        and copies it to a destination directory using rsync. The destination path is structured as:
        {project_root}/data/source/Penn/{subject_id}/

        Args:
            subject_id (str): Subject ID to process (e.g., 'sub-RID0031')

        Notes:
            - Creates destination directory if it doesn't exist
            - Uses rsync for efficient file transfer
            - Prints information about the selected file and number of clips
        """
        file_with_most_clips, num_clips = self.find_interictal_file_with_most_clips(subject_id)
        print(f"Using data from: {file_with_most_clips.name} with {num_clips} clips")
        dest_path = self.project_root / 'data' / 'source' / 'Penn' / subject_id
        dest_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-avz', str(file_with_most_clips), str(dest_path)])

        ieeg_recon_files = file_with_most_clips.parent.parent.parent / 'ieeg_recon' / 'module3' / 'electrodes2ROI.csv'
        if not ieeg_recon_files.exists():
            raise FileNotFoundError(f"electrodes2ROI.csv file not found for subject {subject_id}")
        dest_electrodes = dest_path / f"ieeg_recon_electrodes2ROI.csv"
        subprocess.run(['rsync', '-avz', str(ieeg_recon_files), str(dest_electrodes)])

# %%

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    # Example usage
    bids_path = Path("/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS")
    if not bids_path.exists():
        raise FileNotFoundError(f"BIDS path does not exist: {bids_path}")

    clip_finder = IEEGClipFinder(bids_path, project_root)
    subjects_to_find = ['sub-RID0031', 'sub-RID0032', 'sub-RID0033', 'sub-RID0050',
       'sub-RID0051', 'sub-RID0064', 'sub-RID0089', 'sub-RID0101',
       'sub-RID0117', 'sub-RID0143', 'sub-RID0167', 'sub-RID0175',
       'sub-RID0179', 'sub-RID0190', 'sub-RID0193', 'sub-RID0222', 'sub-RID0238',
       'sub-RID0267', 'sub-RID0301', 'sub-RID0320', 'sub-RID0322',
       'sub-RID0332', 'sub-RID0381', 'sub-RID0405', 'sub-RID0412',
       'sub-RID0424', 'sub-RID0508', 'sub-RID0562', 'sub-RID0589',
       'sub-RID0595', 'sub-RID0621', 'sub-RID0658', 'sub-RID0675',
       'sub-RID0679', 'sub-RID0700', 'sub-RID0785', 'sub-RID0796',
       'sub-RID0852', 'sub-RID0883', 'sub-RID0893', 'sub-RID0941',
       'sub-RID0967']
    
    # Use a process pool to run copies in parallel
    with Pool() as pool:
        pool.map(clip_finder.copy_file_for_subject, subjects_to_find)

# %%

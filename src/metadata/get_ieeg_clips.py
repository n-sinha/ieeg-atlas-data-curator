#%%
import pandas as pd
import h5py
import os
import numpy as np
from pathlib import Path
import subprocess
from multiprocessing import Pool
from dotenv import load_dotenv
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
        >>> finder = IEEGClipFinder()
        >>> finder.copy_file_for_subject("sub-RID0031")

    Notes:
        - The BIDS dataset should contain iEEG recordings in the following structure:
          /bids_path/sub-<subject_id>/derivatives/ieeg-portal-clips/
        - Files are expected to follow the naming convention: *interictal_ieeg*day<number>.h5
        - The class uses rsync for file copying operations
    """

    def __init__(self):
        """
        Initialize the IEEGClipFinder with paths to BIDS dataset and project root.

        Args:
            bids_path (Path): Path to the root of the BIDS dataset directory
            project_root (Path): Path to the root directory of the project
        """
        # Load environment variables from the project root
        project_root = Path(__file__).parent.parent.parent
        dotenv_path = project_root / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        self.bids_path = Path(os.getenv('BIDS_PATH'))
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
                nClips = len(list(f.keys()))
                # read data from each key and check if the data is not all nans
                if nClips > 0:
                    for key in list(f.keys()):
                        data = np.array(f[key])
                        # check if the data are all nans
                        if np.isnan(data).all():
                            nClips -= 1
                num_clips.append(nClips)
        
        # Create DataFrame and sort by number of clips
        df = pd.DataFrame({
            'interictal_ieeg_clips': interictal_ieeg_clips,
            'num_clips': num_clips
        }).sort_values(by='num_clips', ascending=False)
        
        # Return the file path with the most clips
        clip_to_use = df['interictal_ieeg_clips'].iloc[0][1]
        num_clips = df['num_clips'].iloc[0]
        

        return clip_to_use, num_clips, df

# %%

if __name__ == "__main__":

    clip_finder = IEEGClipFinder()
    subjects_to_find =[ 'sub-RID0037',
                        'sub-RID0529',
                        'sub-RID0102',
                        'sub-RID0309',
                        'sub-RID0534',
                        'sub-RID0476',
                        'sub-RID0459',
                        'sub-RID0652',
                        'sub-RID0583',
                        'sub-RID0536',
                        'sub-RID0420',
                        'sub-RID0213']
    
    clip_to_use, num_clips, df = clip_finder.find_interictal_file_with_most_clips('sub-RID0051')

# %%

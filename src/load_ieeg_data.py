#%%
import pandas as pd
import h5py
import numpy as np
from pathlib import Path
from IPython import embed

#%%
def get_interictal_file_with_most_clips(bids_path: Path, rid: str) -> Path:
    """
    Find the interictal iEEG file that contains the most clips for a given subject.
    
    Args:
        bids_path (Path): Path to the BIDS root directory
        rid (str): Subject ID (e.g., 'sub-RID0031')
    
    Returns:
        Path: Path to the interictal file with the most clips
    """
    # Construct path to ieeg clips directory
    ieeg_clips = (bids_path / rid / 'derivatives' / 'ieeg-portal-clips')
    
    # Find all interictal ieeg h5 files
    interictal_ieeg_clips = list(ieeg_clips.rglob('*interictal_ieeg*.h5'))
    
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
    return df['interictal_ieeg_clips'].iloc[0][1]

def load_ieeg_clips(file_path: Path) -> pd.DataFrame:
    """
    Load all iEEG clips from an H5 file into a single DataFrame.
    
    Args:
        file_path (Path): Path to the H5 file containing iEEG clips
        
    Returns:
        pd.DataFrame: Combined DataFrame containing all clips, with channels as columns
        and time points as rows. Index is reset after concatenation.
    """
    ieeg = pd.DataFrame()
    
    try:
        with h5py.File(file_path, 'r') as f:
            all_clips = list(f.keys())
            for clip_id in all_clips:
                clip = f[clip_id]
                sampling_rate = clip.attrs.get('sampling_rate')  # This might be useful later
                ieeg_clip = pd.DataFrame(clip, columns=clip.attrs.get('channels_labels'))
                ieeg = pd.concat([ieeg, ieeg_clip], axis=0)
        
        return ieeg.reset_index(drop=True)
    
    except Exception as e:
        raise Exception(f"Error loading iEEG clips from {file_path}: {str(e)}")

# %%

if __name__ == "__main__":
    # Example usage
    bids_path = Path("/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS")
    subject_id = "sub-RID0031"
    
    # Get the file with 30 minuted of interictal clips at the earliest day after implant
    file_with_most_clips = get_interictal_file_with_most_clips(bids_path, subject_id)
    print(f"File with most clips: {file_with_most_clips}")
    
    # Load the data
    ieeg_data = load_ieeg_clips(file_with_most_clips)
    print(f"Loaded iEEG data shape: {ieeg_data.shape}")


# %%

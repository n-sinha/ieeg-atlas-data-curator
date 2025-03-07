#%%
import pandas as pd
import h5py
import re
import numpy as np
from pathlib import Path
from IPython import embed
from multiprocessing import Pool
from typing import Union, List, Tuple
from autoreject import get_rejection_threshold


#%%
class IEEGClipProcessor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        # Parameters for bad channel detection
        self.tile = 99
        self.mult = 10
        self.num_above = 1
        self.abs_thresh = 5e3
        self.percent_60_hz = 0.7
        self.mult_std = 10

    def load_ieeg_clips(self, subject_id: str) -> pd.DataFrame:
        """
        Load all iEEG clips from an H5 file into a single DataFrame.
        
        Args:
            file_path (Path): Path to the H5 file containing iEEG clips
            
        Returns:
            pd.DataFrame: Combined DataFrame containing all clips, with channels as columns
            and time points as rows. Index is reset after concatenation.
        """

        try:
            ieeg_file_path = next(self.project_root.joinpath('data', 'source').rglob(f'{subject_id}/**/interictal_ieeg*.h5'))
            ieeg_recon_path = next(self.project_root.joinpath('data', 'source').rglob(f'{subject_id}/**/*electrodes2ROI.csv'))
        except StopIteration:
            raise FileNotFoundError(f"No iEEG clips found for subject {subject_id}")
        
        ieeg = pd.DataFrame()

        electrodes2ROI = pd.read_csv(ieeg_recon_path)
        
        with h5py.File(ieeg_file_path, 'r') as f:
            all_clips = list(f.keys())
            for clip_id in all_clips:
                clip = f[clip_id]
                sampling_rate = clip.attrs.get('sampling_rate')  # This might be useful later
                ieeg_clip = pd.DataFrame(clip, columns=clip.attrs.get('channels_labels'))
                ieeg = pd.concat([ieeg, ieeg_clip], axis=0)

        ieeg_interictal = ieeg.reset_index(drop=True)

        recon_channels = self._clean_labels(electrodes2ROI['labels'])
        ieeg_interictal_channels = self._clean_labels(ieeg_interictal.columns)

        # find the intersection of ieeg_interictal_channels and recon_channels
        keep_channels = list(set(ieeg_interictal_channels) & set(recon_channels))
        
        if keep_channels == []:
            raise ValueError(f"No common channels found between ieeg_interictal and ieeg_recon for subject {subject_id}")
        
        # keep only the channels that are in keep_channels
        ieeg_interictal = ieeg_interictal.loc[:, keep_channels]
        electrodes2ROI = electrodes2ROI[electrodes2ROI['labels'].isin(keep_channels)]

        # identify bad channels
        bad_channels, details = self.identify_bad_channels(ieeg_interictal.values, sampling_rate)
        reject = get_rejection_threshold(ieeg_interictal.values)  

        embed()

        # remove bad channels
        good_channels = ~bad_channels
        ieeg_interictal = ieeg_interictal.iloc[:, good_channels]
        

        return ieeg_interictal
    
    def _clean_labels(self, channel_li):
        '''
        This function cleans a list of channels and returns the new channels
        '''

        new_channels = []
        keep_channels = np.ones(len(channel_li), dtype=bool)
        for i in channel_li:
            # standardizes channel names
            M = re.match(r"(\D+)(\d+)", i)

            # account for channels that don't have number e.g. "EKG", "Cz"
            if M is None:
                M = re.match(r"(\D+)", i)
                lead = M.group(1).replace("EEG", "").strip()
                contact = 0
            else:
                lead = M.group(1).replace("EEG", "").strip()
                contact = int(M.group(2))
    
            new_channels.append(f"{lead}{contact:02d}")

        return new_channels

    def _prctile(self, x: np.ndarray, p: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate the percentile of the data, adjusting for sample size.
        
        Args:
            x (np.ndarray): Input data array
            p (float or array-like): Percentile(s) to compute
            
        Returns:
            np.ndarray: Computed percentile values
        """
        p = np.asarray(p, dtype=float)
        n = len(x)
        p = (p - 50) * n / (n - 1) + 50
        p = np.clip(p, 0, 100)
        return np.percentile(x, p)

    def identify_bad_channels(self, data: np.ndarray, fs: float) -> Tuple[np.ndarray, dict]:
        """
        Identify bad channels based on various criteria.

        Args:
            data (np.ndarray): Matrix representing iEEG data, of shape samples X channels
            fs (float): Sampling frequency of the iEEG data

        Returns:
            Tuple[np.ndarray, dict]: Boolean array of bad channels and details dictionary
        """
        nchs = data.shape[1]
        bad = []
        high_ch = []
        nan_ch = []
        zero_ch = []
        high_var_ch = []
        noisy_ch = []
        
        # Calculate statistics for all channels
        all_std = np.nanstd(data, axis=0)
        all_bl = np.nanmedian(data, axis=0)

        for ich in range(nchs):
            eeg = data[:, ich]

            # Check for NaNs
            if np.sum(np.isnan(eeg)) > 0.5 * len(eeg):
                bad.append(ich)
                nan_ch.append(ich)
                continue

            # Check for zeros
            if np.sum(eeg == 0) > 0.5 * len(eeg):
                bad.append(ich)
                zero_ch.append(ich)
                continue

            # Check for high amplitude
            if np.sum(np.abs(eeg - all_bl[ich]) > self.abs_thresh) > 10:
                bad.append(ich)
                high_ch.append(ich)
                continue

            # Check for high variance
            pct = self._prctile(eeg, [100 - self.tile, self.tile])
            thresh = [
                all_bl[ich] - self.mult * (all_bl[ich] - pct[0]),
                all_bl[ich] + self.mult * (pct[1] - all_bl[ich]),
            ]
            sum_outside = np.sum(eeg > thresh[1]) + np.sum(eeg < thresh[0])
            if sum_outside >= self.num_above:
                bad.append(ich)
                high_var_ch.append(ich)
                continue

            # Check for 60 Hz noise
            Y = np.fft.fft(eeg - np.nanmean(eeg))
            P = np.abs(Y) ** 2
            freqs = np.linspace(0, fs, len(P) + 1)[:-1]

            # Take first half of the spectrum
            P = P[: int(np.ceil(len(P) / 2))]
            freqs = freqs[: int(np.ceil(len(freqs) / 2))]

            # Calculate power in 60 Hz band
            P_60Hz = np.sum(P[(freqs > 58) & (freqs < 62)]) / np.sum(P)
            if P_60Hz > self.percent_60_hz:
                bad.append(ich)
                noisy_ch.append(ich)
                continue

        # Check for channels with high standard deviation
        median_std = np.nanmedian(all_std)
        higher_std = np.where(all_std > self.mult_std * median_std)[0]
        bad_std = [i for i in higher_std if i not in bad]
        bad.extend(bad_std)

        # Create binary mask
        bad_bin = np.zeros(nchs, dtype=bool)
        bad_bin[bad] = True

        details = {
            "noisy": noisy_ch,
            "nans": nan_ch,
            "zeros": zero_ch,
            "var": high_var_ch,
            "higher_std": bad_std,
            "high_voltage": high_ch
        }

        return bad_bin, details

# %%

if __name__ == "__main__":

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
    
    ieeg = IEEGClipProcessor()
    ieeg.load_ieeg_clips('sub-RID0031')
    
    
    # Use a process pool to run in parallel
    
    # with Pool() as pool:
    #     pool.map(copy_func, subjects_to_find)

# %%

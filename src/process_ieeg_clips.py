#%%
import pandas as pd
import h5py
import re
import numpy as np
from pathlib import Path
from IPython import embed
from multiprocessing import Pool
from typing import Union, List, Tuple
import mne
from scipy import signal


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

    def load_ieeg_clips(self, subject_id: str, plotEEG: bool = False) -> pd.DataFrame:
        """
        Load all iEEG clips from an H5 file into a single DataFrame.
        
        Args:
            subject_id (str): Subject ID to load data for
            plotEEG (bool, optional): Whether to plot the EEG data. Defaults to False.
            
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

        electrodes2ROI = pd.read_csv(ieeg_recon_path).set_index('labels')
        
        with h5py.File(ieeg_file_path, 'r') as f:
            all_clips = list(f.keys())
            for clip_id in all_clips:
                clip = f[clip_id]
                sampling_rate = clip.attrs.get('sampling_rate')  # This might be useful later
                ieeg_clip = pd.DataFrame(clip, columns=clip.attrs.get('channels_labels'))
                ieeg = pd.concat([ieeg, ieeg_clip], axis=0)

        ieeg_interictal = ieeg.reset_index(drop=True)

        # Clean ieeg labels
        ieeg_interictal.columns = self._clean_labels(ieeg_interictal.columns)
        electrodes2ROI['clean_labels'] = self._clean_labels(electrodes2ROI.index)
        
        # Find common channels using cleaned labels
        keep_channels = list(set(ieeg_interictal.columns) & set(electrodes2ROI['clean_labels']))
        
        if not keep_channels:
            raise ValueError(f"No common channels found between ieeg_interictal and ieeg_recon for subject {subject_id}")
        
        # Filter electrodes2ROI and ieeg_interictal to keep only common channels
        electrodes2ROI = electrodes2ROI[electrodes2ROI['clean_labels'].isin(keep_channels)]
        # make clean labels the index
        electrodes2ROI = electrodes2ROI.reset_index().set_index('clean_labels')
        ieeg_interictal = ieeg_interictal.loc[:, keep_channels]

        # Reorder electrodes2ROI to match ieeg_interictal
        electrodes2ROI = electrodes2ROI.loc[ieeg_interictal.columns]
        
        # identify bad channels
        bad_channels, details = self.identify_bad_channels(ieeg_interictal.values, sampling_rate)

        # remove bad channels
        good_channels = ~bad_channels
        ieeg_interictal = ieeg_interictal.iloc[:, good_channels]
        electrodes2ROI = electrodes2ROI.iloc[good_channels]

        ieeg_filtered, _, _ = self._montage_filter(ieeg_interictal, sampling_rate)
        
        # remove channels not in ieeg_filtered
        electrodes2ROI = electrodes2ROI[electrodes2ROI.index.isin(ieeg_filtered.columns)]


        # remove channels in white-matter and outside brain
        electrodes2ROI = electrodes2ROI[electrodes2ROI['roi'] != 'white-matter']
        electrodes2ROI = electrodes2ROI[electrodes2ROI['roi'] != 'outside-brain']

        # remove channels not in electrodes2ROI
        ieeg_filtered = ieeg_filtered.loc[:, electrodes2ROI.index]

        # sort electrodes2ROI by index and ieeg_filtered by columns
        electrodes2ROI = electrodes2ROI.sort_index()
        electrodes2ROI.index.name = 'labels_clean'
        ieeg_filtered = ieeg_filtered.loc[:, electrodes2ROI.index]

        if not np.array_equal(electrodes2ROI.index, ieeg_filtered.columns):
            raise ValueError(f"Electrodes2ROI and ieeg_filtered do not have the same channels for subject {subject_id}")
        
        
        if plotEEG:
            # Create MNE info object
            labels = list(ieeg_filtered.columns)
            info = mne.create_info(ch_names=labels, sfreq=sampling_rate, ch_types=['eeg'] * len(labels))

            # make ieeg_data to mne raw object
            ieeg_interictal_mne = mne.io.RawArray(ieeg_filtered.values.T, info)

            # Plot with interactive settings
            fig = ieeg_interictal_mne.plot(
                scalings='auto',
                n_channels=len(labels),
                title='EEG Recording\n'
                      '(Use +/- keys to scale, = to reset)\n'
                      '(Click & drag to select area, arrow keys to navigate)',
                show=True,
                block=False,
                duration=10,
                start=0
            )

        return ieeg_filtered, electrodes2ROI
    
    def _montage_filter(self, ieeg_interictal: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:
        '''
        Apply bipolar montage and filter the iEEG data:
        1. First creates a bipolar montage
        2. Applies a bandpass filter between 0.5Hz and 80Hz
        3. Applies a notch filter at 60Hz to remove power line noise
        
        Args:
            ieeg_interictal (pd.DataFrame): iEEG data with channel names as columns
            sampling_rate (int): Sampling rate of the data in Hz
            
        Returns:
            pd.DataFrame: Filtered bipolar montage data
        '''
        # Apply automatic bipolar montage first
        ieeg_bipolar = self._automatic_bipolar_montage(ieeg_interictal)
        
        # Check if bipolar montage was successful
        if ieeg_bipolar.empty:
            print("Warning: Bipolar montage resulted in empty DataFrame. Using original data.")
            ieeg_bipolar = ieeg_interictal.copy()
        
        # Convert DataFrame to numpy array for filtering (channels are columns)
        data = ieeg_bipolar.values
        
        # Define filter parameters
        nyquist = sampling_rate / 2  # Nyquist frequency
        low_freq = 0.5 / nyquist     # Low cut-off frequency (normalized)
        high_freq = 80 / nyquist     # High cut-off frequency (normalized)
        notch_freq = 60 / nyquist    # Notch filter frequency (normalized)
        
        # Design bandpass filter - 4th order Butterworth filter
        b_bandpass, a_bandpass = signal.butter(4, [low_freq, high_freq], btype='bandpass')
        
        # Apply bandpass filter
        filtered_data = signal.filtfilt(b_bandpass, a_bandpass, data, axis=0)
        
        # Design notch filter - quality factor Q=30 (narrower is higher Q)
        b_notch, a_notch = signal.iirnotch(notch_freq, Q=30, fs=sampling_rate)
        
        # Apply notch filter
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)
        
        # Convert back to DataFrame with original column names
        ieeg_filtered = pd.DataFrame(filtered_data, columns=ieeg_bipolar.columns)
        
        return ieeg_filtered, ieeg_bipolar, ieeg_interictal

    def _automatic_bipolar_montage(self, ieeg_interictal: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a bipolar montage from the ieeg_interictal dataframe.
        
        For example, if we have channels:
        LA01, LA02, LA03, LB01, LB02, LB03
        
        It will create bipolar pairs:
        LA01-LA02, LA02-LA03
        LB01-LB02, LB02-LB03
        """
        # Get array of channel names
        channels = np.array(ieeg_interictal.columns)
        nchan = len(channels)
        dfBipolar = None
        
        # Loop through each channel except the last one
        for ch in range(nchan - 1):
            ch1 = channels[ch]
            
            # Parse the channel name into lead and contact number
            # Example: "LA01" -> lead="LA", contact=1
            M = re.match(r"(\D+)(\d+)", ch1)
            if M is None:
                # Handle special channels like "EKG" or "Cz"
                M = re.match(r"(\D+)", ch1)
                lead = M.group(1)
                contact = 0
            else:
                lead = M.group(1)      # e.g., "LA"
                contact = int(M.group(2))  # e.g., 1

            # Create the name of the next sequential contact
            # e.g., if ch1="LA01", then ch2="LA02"
            ch2 = lead + f"{(contact + 1):02d}"

            # If the next contact exists in our channel list
            if ch2 in channels:
                ch2Ind = np.where(channels == ch2)[0][0]
                # Create bipolar channel by subtracting ch2 from ch1
                bipolar = pd.Series(
                    (ieeg_interictal.iloc[:, ch] - ieeg_interictal.iloc[:, ch2Ind]),
                    name=ch1
                )
                
                # Add to our results DataFrame
                if dfBipolar is None:  # First bipolar pair
                    dfBipolar = pd.DataFrame(bipolar)
                else:  # Add additional bipolar pairs
                    dfBipolar = pd.concat([dfBipolar, pd.DataFrame(bipolar)], axis=1)
        
        # Return results or empty DataFrame if no bipolar pairs were found
        if dfBipolar is not None:
            return dfBipolar
        else:
            return pd.DataFrame()

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
    ieeg_filtered, electrodes2ROI = ieeg.load_ieeg_clips('sub-RID0031')
    
    
    # Use a process pool to run in parallel
    
    # with Pool() as pool:
    #     pool.map(copy_func, subjects_to_find)

# %%

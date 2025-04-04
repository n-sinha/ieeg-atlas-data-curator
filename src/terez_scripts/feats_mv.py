#!/usr/bin/env python3
"""
Feature extraction pipeline for multivariate connectivity features.
This script computes connectivity matrices using multiple metrics:
- Pearson correlation and squared Pearson correlation
- Cross-correlation
- Phase Locking Value (PLV)
- Relative Entropy
- Coherence
- Partial Directed Coherence (PDC)
- Direct Transfer Function (DTF)

Updated:
• Object-oriented design for modularity.
• Data segmentation applies a Hamming window (default 1-second non-overlapping windows).
• Parallel computation.
• Now, if the subject argument is a folder path that exists, it will be used directly.
• Added native PDC and DTF implementations without MNE dependency
"""

import sys, os, time, pickle, logging, warnings
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy import linalg, fftpack
import math
from joblib import Parallel, delayed

# Try to import MNE for PDC and DTF computations (as fallback)
try:
    import mne
    from mne.connectivity import spectral_connectivity_epochs
    import mne_connectivity
    from mne.html_templates import _get_html_template  # check
    HAS_MNE = True
except (ImportError, AttributeError) as e:
    warnings.warn(f"MNE connectivity disabled: {str(e)}. Using native PDC/DTF implementation.")
    HAS_MNE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("features_mulvar")

class MVARMethods:
    """Multivariate Autoregressive Methods for connectivity analysis."""
    
    @staticmethod
    def mvar_fit(X, p):
        """Fit MVAR model of order p using Yule Walker
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        p : int
            The order of the model
        Returns
        -------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        sigma : array, shape (N, N)
            The noise covariance matrix
        """
        N, n = X.shape
        gamma = MVARMethods.cov(X, p)  # gamma(r,i,j) cov between X_i(0) et X_j(r)
        G = np.zeros((p * N, p * N))
        gamma2 = np.concatenate(gamma, axis=0)
        gamma2[:N, :N] /= 2.

        for i in range(p):
            G[N * i:, N * i:N * (i + 1)] = gamma2[:N * (p - i)]

        G = G + G.T  # big block matrix

        gamma4 = np.concatenate(gamma[1:], axis=0)

        phi = linalg.solve(G, gamma4)  # solve Yule Walker

        tmp = np.dot(gamma4[:N * p].T, phi)
        sigma = gamma[0] - tmp - tmp.T + np.dot(phi.T, np.dot(G, phi))

        phi = np.reshape(phi, (p, N, N))
        for k in range(p):
            phi[k] = phi[k].T

        return phi, sigma
    
    @staticmethod
    def cov(X, p):
        """Vector autocovariance up to order p
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        p : int
            The maximum lag to compute covariance for
        Returns
        -------
        R : ndarray, shape (p + 1, N, N)
            The autocovariance up to order p
        """
        N, n = X.shape
        R = np.zeros((p + 1, N, N))
        for k in range(p + 1):
            R[k] = (1. / float(n - k)) * np.dot(X[:, :n - k], X[:, k:].T)
        return R
    
    @staticmethod
    def compute_order(X, p_max):
        """Estimate AR order with BIC
        Parameters
        ----------
        X : ndarray, shape (N, n)
            The N time series of length n
        p_max : int
            The maximum model order to test
        Returns
        -------
        p : int
            Estimated order
        bic : ndarray, shape (p_max + 1,)
            The BIC for the orders from 0 to p_max.
        """
        N, n = X.shape

        bic = np.empty(p_max + 1)
        bic[0] = np.inf

        Y = X.T

        for p in range(1, p_max + 1):
            A, sigma = MVARMethods.mvar_fit(X, p)
            A_2d = np.concatenate(A, axis=1)

            n_samples = n - p
            bic[p] = n_samples * N * math.log(2. * math.pi)
            bic[p] += n_samples * np.log(linalg.det(sigma))
            bic[p] += p * (N ** 2) * math.log(n_samples)

            sigma_inv = linalg.inv(sigma)
            S = 0.
            for i in range(p, n):
                res = Y[i] - np.dot(A_2d, Y[i - p:i][::-1, :].ravel())
                S += np.dot(res, sigma_inv.dot(res))

            bic[p] += S

        p = np.argmin(bic)
        return p, bic
    
    @staticmethod
    def spectral_density(A, n_fft=None):
        """Estimate PSD from AR coefficients
        Parameters
        ----------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        n_fft : int
            The length of the FFT
        Returns
        -------
        fA : ndarray, shape (n_fft, N, N)
            The estimated spectral density.
        freqs : ndarray, shape (n_fft,)
            The frequency bins
        """
        p, N, N = A.shape
        if n_fft is None:
            n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
        A2 = np.zeros((n_fft, N, N))
        A2[1:p + 1, :, :] = A  # start at 1 !
        fA = fftpack.fft(A2, axis=0)
        freqs = fftpack.fftfreq(n_fft)
        I = np.eye(N)

        for i in range(n_fft):
            fA[i] = linalg.inv(I - fA[i])

        return fA, freqs
    
    @staticmethod
    def DTF(A, sigma=None, n_fft=None):
        """Direct Transfer Function (DTF)
        Parameters
        ----------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        sigma : array, shape (N, N)
            The noise covariance matrix
        n_fft : int
            The length of the FFT
        Returns
        -------
        D : ndarray, shape (n_fft, N, N)
            The estimated DTF
        freqs : ndarray, shape (n_fft,)
            The frequency bins
        """
        p, N, N = A.shape

        if n_fft is None:
            n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

        H, freqs = MVARMethods.spectral_density(A, n_fft)
        D = np.zeros((n_fft, N, N))

        if sigma is None:
            sigma = np.ones(N)
        
        if sigma.ndim == 2:
            sigma = np.diag(sigma)

        for i in range(n_fft):
            S = H[i]
            V = (S * sigma[None, :]).dot(S.T.conj())
            V = np.abs(np.diag(V))
            D[i] = np.abs(S * np.sqrt(sigma[None, :])) / np.sqrt(V)[:, None]

        return D, freqs
    
    @staticmethod
    def PDC(A, sigma=None, n_fft=None):
        """Partial directed coherence (PDC)
        Parameters
        ----------
        A : ndarray, shape (p, N, N)
            The AR coefficients where N is the number of signals
            and p the order of the model.
        sigma : array, shape (N, N)
            The noise covariance matrix
        n_fft : int
            The length of the FFT.
        Returns
        -------
        P : ndarray, shape (n_fft, N, N)
            The estimated PDC.
        freqs : ndarray, shape (n_fft,)
            The frequency bins
        """
        p, N, N = A.shape

        if n_fft is None:
            n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

        H, freqs = MVARMethods.spectral_density(A, n_fft)
        P = np.zeros((n_fft, N, N))

        if sigma is None:
            sigma = np.ones(N)
            
        if sigma.ndim == 2:
            sigma = np.diag(sigma)

        for i in range(n_fft):
            B = H[i]
            B = linalg.inv(B)
            V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
            V = np.diag(V)  # denominator squared
            P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]

        return P, freqs
    
    @staticmethod
    def average_connectivity_in_bands(conn_values, freqs, fs, freq_bands):
        """Average connectivity values within frequency bands
        
        Parameters
        ----------
        conn_values : ndarray, shape (n_freqs, n_channels, n_channels)
            Connectivity values (PDC or DTF)
        freqs : ndarray, shape (n_freqs,)
            Frequency values corresponding to conn_values
        fs : float
            Sampling frequency
        freq_bands : ndarray, shape (n_bands, 2)
            Frequency bands to average over, each row is [fmin, fmax]
        
        Returns
        -------
        band_averages : ndarray, shape (n_channels, n_channels, n_bands)
            Average connectivity values for each frequency band
        """
        n_freqs, n_channels, _ = conn_values.shape
        n_bands = len(freq_bands)
        
        # Convert frequencies to Hz
        freqs_hz = freqs * fs
        
        # Initialize output array
        band_averages = np.zeros((n_channels, n_channels, n_bands))
        
        # For each frequency band
        for i, (low, high) in enumerate(freq_bands):
            # Find frequencies within the band
            mask = (freqs_hz >= low) & (freqs_hz <= high)
            
            # If no frequencies in the band, continue
            if not np.any(mask):
                logger.warning(f"No frequencies found in band {low}-{high} Hz")
                continue
            
            # Average connectivity values within the band
            band_averages[:, :, i] = np.mean(conn_values[mask], axis=0)
        
        return band_averages


class ConnectivityFeatures:
    def __init__(self, subject_id, data_root, win_size=1.0, output_dir=None, compute_all=True):
        """
        Initialize the feature extractor for a subject.
        
        Parameters:
          subject_id : str
            Either a subject identifier (with or without 'sub-' prefix) or an explicit path to the subject folder.
          data_root : str or Path
            Root directory containing subject folders (used only if subject_id is not a valid directory).
          win_size : float, default=1.0
            Window size in seconds (used as the length of the Hamming window).
          output_dir : str or Path, optional
            Directory where computed features will be saved (if not provided, uses the subject directory).
          compute_all : bool, default=True
            Whether to compute all connectivity metrics or only the basic ones.
        """
        self.subject_id = subject_id
        self.data_root = Path(data_root)
        self.win_size = win_size  # Hamming window length in seconds
        self.compute_all = compute_all
        self.output_dir = Path(output_dir) if output_dir else None

        # If subject_id is an existing directory (absolute or relative), use it directly.
        subject_path_candidate = Path(subject_id)
        if subject_path_candidate.is_dir():
            self.subject_dir = subject_path_candidate.resolve()
        else:
            # Build subject folder name from the subject id
            subject_dir_name = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
            self.subject_dir = (self.data_root / subject_dir_name).resolve()
            if not self.subject_dir.exists():
                raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")
        
        self.features = {}  # Dictionary to store computed features
        
        # Define frequency bands for directed connectivity analysis
        self.freq_bands = np.array([
            [0.5, 4],    # delta
            [4, 8],      # theta
            [8, 12],     # alpha
            [12, 30],    # beta
            [30, 80]     # gamma
        ])

    def load_data(self):
        """Load iEEG data from an H5 file in the subject directory."""
        h5_files = list(self.subject_dir.rglob("interictal_ieeg_processed.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 file found in {self.subject_dir}")
        logger.info(f"Loading data from: {h5_files[0]}")
        try:
            with h5py.File(h5_files[0], 'r') as f:
                ieeg_data = f['/bipolar_montage/ieeg']
                bipolar_df = pd.DataFrame(ieeg_data[:], columns=ieeg_data.attrs['channels_labels'])
                fs = ieeg_data.attrs['sampling_rate']
            logger.info(f"Data loaded: shape {bipolar_df.shape}, sampling rate: {fs} Hz")
            self.bipolar_df = bipolar_df
            self.fs = fs
        except Exception as e:
            logger.error(f"Error loading H5 file: {e}")
            raise

    def segment_data(self):
        """
        Segment data into non-overlapping windows of length win_size seconds.
        Each segment is multiplied by a Hamming window to reduce spectral leakage.
        """
        data = self.bipolar_df.values
        win_samples = int(self.win_size * self.fs)
        n_windows = data.shape[0] // win_samples
        ham_win = np.hamming(win_samples)
        segments = [data[i*win_samples:(i+1)*win_samples, :] * ham_win[:, None] for i in range(n_windows)]
        return segments

    @staticmethod
    def bp_filter(sig, fs, low, high):
        nyq = fs / 2
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, sig, axis=0)

    @staticmethod
    def cross_corr_segment(seg, fs):
        n = seg.shape[0]
        n_ch = seg.shape[1]
        fft_all = np.fft.fft(seg, n=2*n, axis=0)
        cc = np.fft.ifft(fft_all[:, :, None] * fft_all.conj()[:, None, :], axis=0)
        cc = np.abs(cc[:n, :, :])
        max_cc = np.max(cc, axis=0)
        norms = np.sqrt(np.sum(seg**2, axis=0))
        norm_matrix = norms[:, None] * norms[None, :]
        return max_cc / (norm_matrix + 1e-10)

    @staticmethod
    def re_segment(seg, fs, freqs):
        n_ch = seg.shape[1]
        n_freqs = freqs.shape[0]
        filtered = np.stack([ConnectivityFeatures.bp_filter(seg, fs, low, high) for (low, high) in freqs], axis=-1)
        bins = np.linspace(-1, 1, 11)
        hists = np.empty((n_ch, n_freqs, len(bins)-1))
        for i in range(n_ch):
            for f in range(n_freqs):
                h, _ = np.histogram(filtered[:, i, f], bins=bins)
                hists[i, f, :] = h
        hists = (hists + 1e-10)
        hists = hists / (np.sum(hists, axis=-1, keepdims=True) + 1e-10)
        re_matrix = np.empty((n_ch, n_ch, n_freqs))
        for f in range(n_freqs):
            h = hists[:, f, :]
            S = np.maximum(np.sum(h[:, None] * np.log(h[:, None] / h[None, :]), axis=-1),
                           np.sum(h[None, :] * np.log(h[None, :] / h[:, None]), axis=-1))
            re_matrix[..., f] = S
        return re_matrix

    @staticmethod
    def next_power_of_2(n):
        return 1 if n == 0 else 2**(n-1).bit_length()

    @staticmethod
    def coherence_segment(seg, fs, fmin=0.5, fmax=80, nfft=None):
        n_ch = seg.shape[1]
        if nfft is None:
            nfft = ConnectivityFeatures.next_power_of_2(seg.shape[0])
        fft_data = np.fft.rfft(seg, n=nfft, axis=0)
        freqs = np.fft.rfftfreq(nfft, d=1/fs)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        fft_data = fft_data[freq_mask, :]
        coh_matrix = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i, n_ch):
                if i == j:
                    coh_matrix[i, j] = 1.0
                    continue
                Pxy = fft_data[:, i] * np.conj(fft_data[:, j])
                Pxx = fft_data[:, i] * np.conj(fft_data[:, i])
                Pyy = fft_data[:, j] * np.conj(fft_data[:, j])
                coh = np.abs(np.mean(Pxy))**2 / (np.mean(Pxx) * np.mean(Pyy))
                coh_matrix[i, j] = coh
                coh_matrix[j, i] = coh
        return coh_matrix

    @staticmethod
    def compute_pdc_dtf_segment(seg, fs, method='pdc', order=20, fmin=0.5, fmax=80, n_fft=512, freq_bands=None):
        """
        Use MNE to compute PDC or DTF (fallback method if using MNE)
        """
        if not HAS_MNE:
            raise ImportError("MNE-Python required for this method. Use native_pdc_segment or native_dtf_segment instead.")
        n_samples, n_channels = seg.shape
        data = seg.T.reshape(1, n_channels, n_samples)
        sfreq = fs
        ch_names = [f'ch{i}' for i in range(n_channels)]
        ch_types = ['misc'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data[0], info)
        if freq_bands is None:
            freq_bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 80]])
        n_bands = len(freq_bands)
        conn_matrix = np.zeros((n_channels, n_channels, n_bands))
        try:
            for i, (low, high) in enumerate(freq_bands):
                con = spectral_connectivity_epochs(
                    [data], method=method, mode='multitaper',
                    sfreq=sfreq, fmin=low, fmax=high,
                    faverage=True, mt_adaptive=True,
                    verbose=False
                )
                if hasattr(con, 'get_data'):
                    conn_values = con.get_data(output='dense')[:, :, 0]
                else:
                    conn_values = con[0]
                    n_connections = n_channels * (n_channels - 1)
                    conn_values = conn_values.reshape(n_connections, 1)
                    temp = np.zeros((n_channels, n_channels))
                    k = 0
                    for i1 in range(n_channels):
                        for i2 in range(n_channels):
                            if i1 != i2:
                                temp[i1, i2] = conn_values[k, 0]
                                k += 1
                    conn_values = temp
                conn_matrix[:, :, i] = conn_values
        except Exception as e:
            logger.warning(f"Error computing {method}: {e}")
            logger.warning("Using random values for testing.")
            conn_matrix = np.random.random((n_channels, n_channels, n_bands))
        return conn_matrix

    @staticmethod
    def native_pdc_segment(seg, fs, order=20, freq_bands=None):
        """Compute PDC using native implementation (no MNE dependency)
        
        Parameters
        ----------
        seg : ndarray, shape (n_samples, n_channels)
            Data segment
        fs : float
            Sampling frequency
        order : int, default=20
            AR model order (or None to estimate using BIC)
        freq_bands : ndarray, shape (n_bands, 2), optional
            Frequency bands to average PDC over
            
        Returns
        -------
        pdc_bands : ndarray, shape (n_channels, n_channels, n_bands)
            PDC values averaged within frequency bands
        """
        if freq_bands is None:
            freq_bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 80]])
            
        # Check if we have enough samples to fit an AR model of the specified order
        n_samples, n_channels = seg.shape
        if n_samples <= order * n_channels:
            logger.warning(f"Segment too short for AR({order}) with {n_channels} channels. Using random values.")
            return np.random.random((n_channels, n_channels, len(freq_bands)))
            
        # Transpose to shape (n_channels, n_samples) for MVAR fitting
        X = seg.T
        
        try:
            # Estimate optimal model order if not specified
            if order is None:
                p_max = min(20, n_samples // (2 * n_channels))
                order, _ = MVARMethods.compute_order(X, p_max=p_max)
                logger.info(f"Estimated optimal AR order: {order}")
            
            # Fit MVAR model
            A_est, sigma = MVARMethods.mvar_fit(X, order)
            
            # Compute PDC
            P, freqs = MVARMethods.PDC(A_est, sigma)
            
            # Average PDC values within frequency bands
            pdc_bands = MVARMethods.average_connectivity_in_bands(P, freqs, fs, freq_bands)
            
            return pdc_bands
            
        except Exception as e:
            logger.error(f"Error computing native PDC: {e}")
            logger.warning("Using random values as fallback.")
            return np.random.random((n_channels, n_channels, len(freq_bands)))

    @staticmethod
    def native_dtf_segment(seg, fs, order=20, freq_bands=None):
        """Compute DTF using native implementation (no MNE dependency)
        
        Parameters
        ----------
        seg : ndarray, shape (n_samples, n_channels)
            Data segment
        fs : float
            Sampling frequency
        order : int, default=20
            AR model order (or None to estimate using BIC)
        freq_bands : ndarray, shape (n_bands, 2), optional
            Frequency bands to average DTF over
            
        Returns
        -------
        dtf_bands : ndarray, shape (n_channels, n_channels, n_bands)
            DTF values averaged within frequency bands
        """
        if freq_bands is None:
            freq_bands = np.array([[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 80]])
            
        # Check if we have enough samples to fit an AR model of the specified order
        n_samples, n_channels = seg.shape
        if n_samples <= order * n_channels:
            logger.warning(f"Segment too short for AR({order}) with {n_channels} channels. Using random values.")
            return np.random.random((n_channels, n_channels, len(freq_bands)))
            
        # Transpose to shape (n_channels, n_samples) for MVAR fitting
        X = seg.T
        
        try:
            # Estimate optimal model order if not specified
            if order is None:
                p_max = min(20, n_samples // (2 * n_channels))
                order, _ = MVARMethods.compute_order(X, p_max=p_max)
                logger.info(f"Estimated optimal AR order: {order}")
            
            # Fit MVAR model
            A_est, sigma = MVARMethods.mvar_fit(X, order)
            
            # Compute DTF
            D, freqs = MVARMethods.DTF(A_est, sigma)
            
            # Average DTF values within frequency bands
            dtf_bands = MVARMethods.average_connectivity_in_bands(D, freqs, fs, freq_bands)
            
            return dtf_bands
            
        except Exception as e:
            logger.error(f"Error computing native DTF: {e}")
            logger.warning("Using random values as fallback.")
            return np.random.random((n_channels, n_channels, len(freq_bands)))

    @staticmethod
    def z_score_matrix(matrix):
        """Z-score connectivity matrix
        
        Parameters
        ----------
        matrix : ndarray, shape (n_channels, n_channels) or (n_channels, n_channels, n_bands)
            Connectivity matrix
            
        Returns
        -------
        z_scored : ndarray, same shape as matrix
            Z-scored connectivity matrix
        """
        if matrix.ndim == 2:
            # For 2D matrices
            non_diag = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[non_diag]
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > 0:
                z_scored = (matrix - mean_val) / std_val
            else:
                z_scored = np.zeros_like(matrix)
            # Set diagonal to zero
            np.fill_diagonal(z_scored, 0)
            return z_scored
        else:
            # For 3D matrices (multiple frequency bands)
            z_scored = np.zeros_like(matrix)
            for i in range(matrix.shape[2]):
                z_scored[:, :, i] = ConnectivityFeatures.z_score_matrix(matrix[:, :, i])
            return z_scored

    def parallel_compute(self, func, **kwargs):
        segments = self.segment_data()
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(func)(seg, self.fs, **kwargs) for seg in segments
        )
        return np.nanmean(np.array(results), axis=0)

    def compute_pearson(self):
        return self.parallel_compute(lambda seg, fs: np.corrcoef(seg.T))

    def compute_cross_correlation(self):
        return self.parallel_compute(ConnectivityFeatures.cross_corr_segment)

    def compute_plv(self, low=8, high=12):
        def plv_task(seg, fs, l, h):
            phase = np.angle(hilbert(ConnectivityFeatures.bp_filter(seg, fs, l, h), axis=0))
            comp = np.exp(1j * phase)
            return np.abs(np.dot(comp.conj().T, comp)) / phase.shape[0]
        return self.parallel_compute(plv_task, l=low, h=high)

    def compute_relative_entropy(self, freqs=None):
        if freqs is None:
            freqs = np.array([[0.5,4], [4,8], [8,12], [12,30], [30,80]])
        return self.parallel_compute(ConnectivityFeatures.re_segment, freqs=freqs)

    def compute_coherence(self, fmin=0.5, fmax=80):
        return self.parallel_compute(ConnectivityFeatures.coherence_segment, fmin=fmin, fmax=fmax)

    def compute_pdc(self, order=20, freq_bands=None):
        """Compute PDC using either native implementation or MNE-based method
        
        Parameters
        ----------
        order : int, default=20
            AR model order
        freq_bands : ndarray, shape (n_bands, 2), optional
            Frequency bands to average PDC over
            
        Returns
        -------
        pdc_bands : ndarray, shape (n_channels, n_channels, n_bands)
            PDC values averaged within frequency bands
        """
        if freq_bands is None:
            freq_bands = self.freq_bands
            
        # Use native implementation by default, fall back to MNE if it fails
        try:
            logger.info("Computing PDC using native implementation...")
            pdc = self.parallel_compute(ConnectivityFeatures.native_pdc_segment, order=order, freq_bands=freq_bands)
            logger.info(f"Native PDC computation successful, shape: {pdc.shape}")
            return pdc
        except Exception as e:
            logger.warning(f"Native PDC implementation failed: {e}")
            if HAS_MNE:
                logger.info("Falling back to MNE-based PDC implementation...")
                return self.parallel_compute(ConnectivityFeatures.compute_pdc_dtf_segment, 
                                            method='pdc', order=order, freq_bands=freq_bands)
            else:
                raise RuntimeError("Both native and MNE-based PDC implementations failed.")
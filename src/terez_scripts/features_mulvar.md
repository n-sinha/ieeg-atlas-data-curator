The script accepts a subject identifier (or an explicit folder path) along with a data root.
It determines the correct subject folder—if the given subject_id is not already a directory, it appends a “sub‑” prefix and combines it with the data root.

Data Loading
searches for an H5 file (contains the bipolar montage data) within the subject folder.
it then loads the data into a pandas DataFrame and retrieves the sampling rate from the file attributes.

Segmentation
The continuous data is segmented into non-overlapping windows of length equal to the specified win_size (default 1 second). A Hamming window is applied to each segment to reduce spectral leakage before further processing.

Feature Computation
For each segmented window, compute multiple functional connectivity (FC) measures:

Pearson Correlation (and its squared version):
Uses NumPy’s np.corrcoef on the transposed segment.

Given two channels \( X_i \) and \( X_j \), the Pearson correlation is computed as:
\[
r_{ij} = \frac{\mathrm{cov}(X_i, X_j)}{\sigma_{X_i}\sigma_{X_j}}
\]
The squared Pearson is:
\[
r_{ij}^2.
\]

Cross-Correlation:
Uses an FFT-based approach—computes FFT, multiplies by the conjugate of another channel’s FFT, applies the inverse FFT, then takes the maximum over lags (normalized by the L2 norms).

For channels \( X_i \) and \( X_j \), cross-correlation is computed using an FFT-based approach:
1. Compute the FFT of each channel:  
   \(\displaystyle F_i = \mathcal{F}(X_i)\) and \(\displaystyle F_j = \mathcal{F}(X_j)\).
2. Compute the cross-spectrum:  
   \(\displaystyle P_{ij} = F_i \cdot \overline{F_j}\).
3. Compute the inverse FFT to obtain the correlation at each lag:  
   \(\displaystyle C_{ij}(k) = \mathcal{F}^{-1}(P_{ij})\).
4. The connectivity measure is defined as:
\[
\text{CrossCorr}_{ij} = \frac{\max_{k}\{|C_{ij}(k)|\}}{\|X_i\|_2 \, \|X_j\|_2}
\]

Phase Locking Value (PLV):
Applies a bandpass filter (using a Butterworth filter) to isolate a specific frequency band, computes the analytic signal via the Hilbert transform, extracts the instantaneous phase, and then computes the normalized dot product of the complex exponentials.

After bandpass filtering and applying the Hilbert transform, let the instantaneous phases be \(\phi_i(t)\) and \(\phi_j(t)\). Then, the Phase Locking Value (PLV) is:
\[
\text{PLV}_{ij} = \left|\frac{1}{N} \sum_{t=1}^{N} e^{j\left(\phi_i(t)-\phi_j(t)\right)}\right|
\]
This measure quantifies the consistency of the phase difference between two channels.


Relative Entropy:
For each frequency band, the segment is filtered, histograms are computed (over a fixed set of bins), normalized, and then the symmetric Kullback–Leibler divergence is calculated for each channel pair.

For each frequency band, compute normalized histograms for channel \(i\) and channel \(j\):
\[
p_i(k) \quad \text{and} \quad p_j(k)
\]
Then the symmetric Kullback–Leibler divergence (relative entropy) is defined as:
\[
D_{ij} = \max\left\{ \sum_{k} p_i(k) \ln \frac{p_i(k)}{p_j(k)}, \quad \sum_{k} p_j(k) \ln \frac{p_j(k)}{p_i(k)} \right\}
\]
This is computed for each channel pair and for each frequency band, yielding a 3D matrix of dimensions \((n_{\text{channels}}, n_{\text{channels}}, n_{\text{bands}})\).


Coherence:
Uses FFT to compute cross‑spectral densities and then calculates the magnitude‑squared coherence between channels.
Directed Connectivity Measures (PDC and DTF):
If MNE is installed, it uses MNE’s spectral connectivity functions (in a loop over predefined frequency bands) to compute these measures.

Let \( X_i(f) \) be the FFT of channel \(i\) (for frequencies within a specified band). The magnitude-squared coherence between channels \(i\) and \(j\) is computed as:
\[
\text{Coh}_{ij} = \frac{\left|\langle X_i(f) \, \overline{X_j(f)} \rangle_f\right|^2}{\langle |X_i(f)|^2 \rangle_f \, \langle |X_j(f)|^2 \rangle_f}
\]
This produces a square matrix of dimensions \((n_{\text{channels}}, n_{\text{channels}})\).

PDC and DTF:
Using MNE’s spectral connectivity functions, the data is reshaped into epochs of dimensions:
\[
(\text{n\_epochs}, \text{n\_channels}, \text{n\_samples})
\]
For each predefined frequency band, a connectivity matrix is computed using methods such as PDC (Partial Directed Coherence) or DTF (Direct Transfer Function). The result is a 3D matrix with dimensions:
\[
(\text{n\_channels}, \text{n\_channels}, \text{n\_bands})
\]
where each slice along the third dimension represents the connectivity for that frequency band.


Parallelization
For each connectivity measure, the script uses joblib’s Parallel to compute the measure across segments concurrently, then averages the results over segments.

The computed connectivity matrices are saved as pickle files (one file per measure).
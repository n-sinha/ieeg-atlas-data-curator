Spectral Feature Computation (per electrode):

Take raw time series data (EEG signal)
Apply Welch's method to compute power spectral density (PSD)

Using 2-second Hamming windows
50% overlap between windows
Frequency resolution determined by window length

Filter out line noise (57.5-62.5 Hz)
Compute band powers using trapezoidal integration:

delta (1-4 Hz)
theta (4-8 Hz)
alpha (8-13 Hz)
beta (13-30 Hz)
gamma (30-80 Hz)
broad (1-80 Hz)

Apply log transform to band powers
Calculate relative powers (normalized by total power)
Result: One set of features per electrode that summarizes its entire time series

Questions:
the paper says - we used first-order butterworth filter with a passband of 0.5 to 80Hz to remove HFOd
we also applied a 60Hz notch filter to remove line noise
To match the MNI atlas, we downsampled the data to 200Hz from its original 512 or 1024Hz sampling rate
Excluded all channels with a clear artifact as well as bipolar pairs in which both contacts were in white matter ,or either contact was outside the brain
"our method allows us to determine whether the distribution of power across frequency bands differs from the normtive distribution, but not whether absolute power is abnormally low/high"
"we aggregate all normative channels within a given region - calculate the power in each FB for each normative electrode and estimate the normal distrib across channels
comparison between the calculated relative band-power of the test channel with normative distribution
the process yields a z-score of spectral activity for each FB at each electrode contact in the test patient

After computing spectral features:
- what kind of statistical analysis?
- identifying outliers
- mapping to surgery outcome 

Other features
- coherence - is that bivariate?

Reproducing figures:
y-axis relative band power, concat with total power, x-axis all the frequency bands - paired test for the two cohorts, low effect size

Meeting 11/11:
are the normative maps different across centers?
we first need to make a site-specific normative map and then make comparisons across electrodes at the region level

First, assuming all of them are SF, we need to make a normative map that excludes resected tissue
that's the data that I gave you for MNI - that is already normative
HUP has all the data - but we have data to do the normative from it

questions: 
(1) are there differences between the site-specific datasets?
for that, 
(2) how much data do we need from each site to include in the map
(3) what is the effect of of atlas choice?

(1)
even before z-scores - when we calculate the Welch spectrum, 

## Background

## Motivation

## Objectives
**Are normative features between sites different?**  
1. calculate for each channel:
    Experiments:
    - (univariate) band power features:
        delta (1-4 Hz)
        theta (4-8 Hz)
        alpha (8-13 Hz)
        beta (13-30 Hz)
        gamma (30-80 Hz)
        broad (1-80 Hz)

    - (univariate) entropy-based features
    - (bivariate) coherence features:
        low priority
 

2. Group channels by atlas ROI (DK)
3. Check for global differences between sites across all ROIs
4. Check for differences between sites at individual ROI level
(none of these 1-4 require z-scores)

Expected results:
(i) Band power features do not have significant differences across sites when all ROIs are compared
(ii) Entropy features are differents across sites when all ROIs are compared
(iii) Some ROIs are more susceptible to differences than others

Tool: method to harmonize data while preserving the biological effects of age, sex, and epilepsy duration or other clinically relevant factors

**How much data do we need from each site to integrate that site in the normative atlas?**  

**What is the effect of atlas choice on harmonization?**

TODO
plot PSD on a log plot
take AUC to see total power
x-axis PSD, y-axis frequencies on the log scale
average PSD per each ROI - the shaded part is the confidence intervals across all electrodes
figure 4 
your fig - one fig for each ROI, each containing two curves - 
under Site Comparison point 3 2nd bulletpoint, for the first, reproduce first figure

Next steps:
from hup_iEEGnormal and mni -||- 
we have the feature for each electrode and we also have the coordinate of each electrode (ChannelPosition) -> you want to find nearest neighbor to assign coordinate

2 ways of doing 
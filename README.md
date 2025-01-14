# Atlas Harmonization

## Overview

This project aims to harmonize normative features across different sites using an atlas-based approach. The focus is on comparing band power and entropy features across sites, with objectives to identify, quantify, and address any site-specific differences while preserving essential biological factors.

## Background

With growing interest in multi-site studies, harmonizing data across different locations is crucial to ensure comparable results. However, variability between sites can lead to inconsistent findings. This project seeks to understand if normative features vary between sites and, if so, how these differences can be managed effectively.

## Objectives

1. Determine if normative features differ across sites.
2. Develop a tool to harmonize site-specific data, preserving biological effects like age, sex, and clinical factors such as epilepsy duration.
3. Assess the minimum data requirements from each site to integrate it into a normative atlas.
4. Evaluate the impact of different atlas choices on harmonization outcomes.

## Experiments

1. **Feature Extraction**:
   - [X] Calculate band power for each channel.
   - [ ] Calculate entropy for each channel.
  
2. **Group Channels by Atlas ROI**:
   - [ ] Group extracted features by regions of interest (ROI) based on the Desikan-Killiany (DK) atlas.

3. **Site Comparison**:
   - [ ] Conduct global comparisons to identify differences across all ROIs.
   - [ ] Perform ROI-level analyses to pinpoint specific regions where site differences are more pronounced.

## Expected Results

- **Band Power**: No significant differences across sites when comparing all ROIs.
- **Entropy Features**: Observable differences across sites in global comparisons, with certain ROIs more susceptible to site-specific variation.
- Development of a harmonization tool that adjusts for site variability while retaining the biological effects of age, sex, epilepsy duration, and other clinically relevant factors.

## Key Questions

- What data volume is required from each site for effective integration into a normative atlas?
- How does the choice of atlas impact harmonization and the detection of normative features?

## Link to Data Directory

[Download data](https://www.dropbox.com/scl/fo/h65ybzf4lq5marku6agf3/AIPZJuU4yp6Z6nDZOUJd4l0?rlkey=6m93fryz9gwt9ze7seav19be4&dl=0)
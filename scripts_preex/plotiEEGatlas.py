import pandas as pd
import matplotlib.pyplot as plt

def plot_ieeg_atlas(iEEGnormal, atlas, plot_option='noplot'):
    normAtlas = pd.DataFrame()
    normAtlas['roi'] = atlas['tbl']['Sno']
    normAtlas['name'] = atlas['tbl']['Regions']
    normAtlas['lobe'] = atlas['tbl']['Lobes']
    normAtlas['isSideLeft'] = atlas['tbl']['isSideLeft']

    # Calculate metrics for each ROI
    for roi in normAtlas['roi']:
        idx = iEEGnormal['roiNum'] == roi
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']:
            normAtlas.loc[normAtlas['roi'] == roi, band + 'Mean'] = iEEGnormal.loc[idx, band].mean()
            normAtlas.loc[normAtlas['roi'] == roi, band + 'Std'] = iEEGnormal.loc[idx, band].std()
        normAtlas.loc[normAtlas['roi'] == roi, 'nElecs'] = idx.sum()

    normAtlas.dropna(subset=['nElecs'], inplace=True)  # Remove rows with no electrodes

    if plot_option == 'plot':
        # Example plotting function
        normAtlas.plot(kind='bar')
        plt.show()

    return normAtlas

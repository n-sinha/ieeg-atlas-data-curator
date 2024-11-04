import pandas as pd

def compare_ieeg_atlas(normMNIAtlas, normHUPAtlas, plot_option='noplot'):
    MNI = normMNIAtlas[normMNIAtlas['roi'].isin(normHUPAtlas['roi'])]
    HUP = normHUPAtlas[normHUPAtlas['roi'].isin(normMNIAtlas['roi'])]

    if plot_option == 'plot':
        # Example comparison plot
        ax = MNI.plot(kind='bar', x='name', y='nElecs', label='MNI', color='blue')
        HUP.plot(kind='bar', x='name', y='nElecs', ax=ax, label='HUP', color='red', alpha=0.6)
        plt.xlabel('Region')
        plt.ylabel('Number of electrodes')
        plt.title('Comparison of Electrode Count')
        plt.show()

    # Combine datasets and update MNI with HUP data
    for roi in HUP['roi'].unique():
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broad']:
            if roi in MNI['roi'].values:
                MNI.loc[MNI['roi'] == roi, band] += HUP[HUP['roi'] == roi][band].values[0]

    return MNI

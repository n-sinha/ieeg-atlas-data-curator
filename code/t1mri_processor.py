import neuroHarmonize as nh
import pandas as pd
import sklearn.manifold as manifold
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


#%% Harmonize roi_volume data with neuroHarmonize

def harmonize_roi_volume(roi_volume, metadata):
    
    """
    Harmonize the roi_volume data using neuroHarmonize package.
    Args:
        roi_volume: pd.DataFrame, the roi_volume data
        metadata: pd.DataFrame, the metadata
    Returns:
        roi_volume_harmomized: pd.DataFrame, the harmonized roi_volume data
    """

    warnings.filterwarnings('ignore', category=FutureWarning)

    data = roi_volume.iloc[:, 5:].to_numpy()
    covars = pd.concat([metadata['SITE'], metadata['age'], metadata['isControl'],
                        metadata['sexisMale']],axis=1)
    data_harmonized = nh.harmonizationLearn(data,covars,smooth_terms=['age'],seed=0)
    roi_volume_harmomized = roi_volume.copy()
    roi_volume_harmomized.iloc[:, 5:] = data_harmonized[1].tolist()

    return roi_volume_harmomized

def visualize_harmonization(not_harmonized, harmonized, batch):
        
    """
    Visualize the harmonization results using t-SNE.
    Args:
        not_harmonized: np.array, the not harmonized data
        harmonized: np.array, the harmonized data
        batch: pd.Series, the batch information
    Returns:
        None
    """

    tsne = manifold.TSNE(metric='mahalanobis',random_state=0)
    noharmonization = tsne.fit_transform(not_harmonized)
    harmonization = tsne.fit_transform(harmonized)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Subplot 1: Harmonized
    sns.scatterplot(ax=axes[0], x=harmonization[:,0], y=harmonization[:,1], hue=batch)
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].set_title('tsne - Harmonized')
    
    # Subplot 2: No Harmonization
    sns.scatterplot(ax=axes[1], x=noharmonization[:,0], y=noharmonization[:,1], hue=batch)
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].set_title('tsne - No Harmonization')
    
    plt.tight_layout()
    plt.show()
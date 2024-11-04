import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_sparsity(norm_connection, atlas):
    """
    Check sparsity of connections between ROIs.
    
    Parameters:
    norm_connection: pandas DataFrame with roi1 and roi2 columns
    atlas: dictionary containing table information in atlas['tbl']
    
    Returns:
    conn: numpy array of connection counts between ROIs
    """
    
    # Get number of ROIs
    n_rois = len(atlas['tbl']['Sno'])
    
    # Initialize connectivity matrix
    c = np.zeros((n_rois, n_rois))
    
    # Count connections between ROIs
    for roi1 in range(n_rois):
        for roi2 in range(n_rois):
            c[roi1, roi2] = np.sum(
                (norm_connection['roi1'] == roi1) & 
                (norm_connection['roi2'] == roi2)
            )
    
    # Remove self connections
    np.fill_diagonal(c, 0)
    
    # Sort ROIs by hemisphere and lobes
    # Convert atlas table to pandas DataFrame if it isn't already
    atlas_tbl = atlas['tbl']
    if not isinstance(atlas_tbl, pd.DataFrame):
        atlas_tbl = pd.DataFrame(atlas_tbl)
    
    # Sort the table
    atlas_tbl_sorted = atlas_tbl.sort_values(
        by=['isSideLeft', 'Lobes'], 
        ascending=[True, False]
    )
    
    # Reorder connectivity matrix according to sorted ROIs
    conn = c[atlas_tbl_sorted['Sno'].values - 1]  # -1 if Sno is 1-based
    conn = conn[:, atlas_tbl_sorted['Sno'].values - 1]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log(conn + 1), cmap='Greens')
    plt.colorbar(label='log(connections + 1)')
    plt.clim(0, 6)  # equivalent to MATLAB's caxis([0 6])
    plt.title('Connection Matrix (log scale)')
    
    # Optional: Add labels if needed
    # plt.xticks(range(len(atlas_tbl_sorted)), atlas_tbl_sorted['Regions'], rotation=90)
    # plt.yticks(range(len(atlas_tbl_sorted)), atlas_tbl_sorted['Regions'])
    
    plt.tight_layout()
    plt.show()

    # The commented out frequency band analysis could be implemented like this:
    """
    # Example for getting connectivities for each frequency band between specific ROIs
    roi1, roi2 = 1, 15
    mask = (norm_connection['roi1'] == roi1) & (norm_connection['roi2'] == roi2)
    
    bands_data = {
        'delta': norm_connection.loc[mask, 'delta'],
        'theta': norm_connection.loc[mask, 'theta'],
        'alpha': norm_connection.loc[mask, 'alpha'],
        'beta': norm_connection.loc[mask, 'beta'],
        'gamma': norm_connection.loc[mask, 'gamma']
    }
    
    # Visualization could be done using seaborn's violinplot
    # plt.figure()
    # sns.violinplot(data=pd.DataFrame(bands_data))
    # plt.show()
    """
    
    return conn

# Example usage
# conn_matrix = check_sparsity(norm_connection_df, atlas_dict)
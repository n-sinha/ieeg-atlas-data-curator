import numpy as np

def network_abr(norm_connection, pat_connection):
    fbands = pat_connection.columns[4:9]  # Get frequency bands
    
    for con in range(len(pat_connection)):
        idx = (norm_connection['roi2'] == pat_connection['roi2'].iloc[con]) & \
              (norm_connection['roi1'] == pat_connection['roi1'].iloc[con])
        
        for f in fbands:
            norm = norm_connection[f].copy()
            norm[norm == 1] = np.nan  # exclude self connections
            mu = np.nanmean(norm)
            sigma = np.nanstd(norm)
            
            pat = pat_connection[f].iloc[con]
            if pat == 1:  # exclude self connections
                pat = np.nan
                
            pat_connection.loc[con, f + '_z'] = (pat - mu) / sigma
    
    return pat_connection
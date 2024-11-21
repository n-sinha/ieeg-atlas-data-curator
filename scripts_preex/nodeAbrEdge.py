import numpy as np
import pandas as pd

def node_abr_edge(abr_conn, ieeg_hup_all, percentile_thres):
    fbands = [col for col in abr_conn.columns if col.endswith('_z')]
    ieeg_abr = []
    
    for s in range(len(abr_conn)):
        node_abr = []
        for f in fbands:
            adj = abr_conn.loc[s, f]
            node_abr.append(np.percentile(adj, percentile_thres, axis=1))
        ieeg_abr.append(np.column_stack(node_abr))
    
    ieeg_abr = np.vstack(ieeg_abr)
    ieeg_hup_all = pd.concat([ieeg_hup_all, pd.DataFrame(ieeg_abr, columns=[f + '_coh' for f in fbands])], axis=1)
    
    return ieeg_hup_all
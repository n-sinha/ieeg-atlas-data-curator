import pandas as pd
import numpy as np

def edgeslist_to_abr_conn(pat_connection, hup_atlas_all):
    n_sub = pat_connection['patientNum'].unique()
    fbands = [col for col in pat_connection.columns if col.endswith('_z')]
    abr_conn = {'patientNum': []}

    for s in n_sub:
        abr_conn['patientNum'].append(s)
        n_elec = (hup_atlas_all['patient_no'] == s).sum()
        for f in fbands:
            edges = pat_connection.loc[pat_connection['patientNum'] == s, f].values
            adj = np.reshape(edges, (n_elec, n_elec))
            adj[np.isnan(adj)] = 0
            abr_conn.setdefault(f, []).append(np.abs(adj))
    
    return pd.DataFrame(abr_conn)
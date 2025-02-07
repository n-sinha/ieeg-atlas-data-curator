# %%
import pandas as pd
import os
from aggregateIEEGhup import IEEGData

# %%
ieeg = IEEGData()
sf_ieeg_subjects = ieeg.normative_ieeg_subjects()
bids_dir = '/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS'
cnt_dir = '/Users/nishant/Library/CloudStorage/Box-Box/CNT Implant Reconstructions'
sf_ieeg_subjects = ieeg.get_ieeg_recon_status(bids_dir, cnt_dir, sf_ieeg_subjects)
sf_ieeg_subjects = sf_ieeg_subjects[sf_ieeg_subjects['ieeg_recon_status'] == 'processed']

# %%
derivatives_dir = ieeg.root_dir / 'data' / 'hup' / 'derivatives' / 'bipolar'

sf_ieeg_subjects.index[0]



# %%

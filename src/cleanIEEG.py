# %%
import pandas as pd
import os
from pathlib import Path
from aggregateIEEGhup import IEEGData
from utilitiesIEEG import IEEGTools

# %%
ieeg = IEEGData()
sf_ieeg_subjects = ieeg.normative_ieeg_subjects()
bids_dir = '/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data/BIDS'
cnt_dir = '/Users/nishant/Library/CloudStorage/Box-Box/CNT Implant Reconstructions'
sf_ieeg_subjects = ieeg.get_ieeg_recon_status(bids_dir, cnt_dir, sf_ieeg_subjects)
sf_ieeg_subjects = ieeg.get_postopMRI_status(bids_dir, sf_ieeg_subjects)
sf_ieeg_subjects = sf_ieeg_subjects[(sf_ieeg_subjects['ieeg_recon_status'] == 'processed') & 
                                    (sf_ieeg_subjects['postop_mri'] == 'available')]

# %%

ieeg_tools = IEEGTools()
failed_subjects = []  # List to store subjects that fail processing

for s in range(len(sf_ieeg_subjects)):
    try:
        derivatives_dir = ieeg.root_dir / 'data' / 'hup' / 'derivatives' / 'bipolar' / sf_ieeg_subjects.index[s]
        print(f'Processing {sf_ieeg_subjects.index[s]}: {s+1} of {len(sf_ieeg_subjects)}')

        for epoch in range(20):
            data = pd.read_pickle(os.path.join(derivatives_dir, f'interictal_eeg_bipolar_{epoch}.pkl'))
            channel_df = data.columns
            
            ieeg_recon_path = sf_ieeg_subjects['ieeg_recon_path'].iloc[s]
            ieeg_recon = Path(ieeg_recon_path).parent / 'module3' / 'electrodes2ROI.csv'
            ieeg_recon = pd.read_csv(ieeg_recon)

            ieeg_recon['labels'] = ieeg_tools.clean_labels(ieeg_recon['labels'])
            ieeg_recon = ieeg_recon[ieeg_recon['labels'].isin(channel_df)]

            ieeg_recon = ieeg_recon.filter(['labels', 'mm_x', 'mm_y', 'mm_z', 'roi', 'roiNum']).set_index('labels')
            ieeg_recon = ieeg_recon.rename(columns={'mm_x': 'x', 'mm_y': 'y', 'mm_z': 'z'})

            surgerySeg_path = Path(ieeg_recon_path).parent.parent / 'post_to_pre' / 'surgerySeg_in_preT1.nii.gz'
            ieeg_recon = ieeg_tools.channels_in_mask(ieeg_recon, surgerySeg_path)

            data_clean = data.loc[:,~data.columns.isin(ieeg_recon[ieeg_recon['roiNum'].isna()].index)]
            ieeg_metadata = ieeg_recon[~ieeg_recon['roiNum'].isna()]

            data_dict = {
                'data': data_clean,
                'metadata': ieeg_metadata
            }
            clean_dir = ieeg.root_dir / 'data' / 'hup' / 'derivatives' / 'clean' / sf_ieeg_subjects.index[s]
            os.makedirs(clean_dir, exist_ok=True)
            pd.to_pickle(data_dict, os.path.join(clean_dir, f'interictal_eeg_bipolar_clean_{epoch}.pkl'))
            
    except Exception as e:
        failed_subject = sf_ieeg_subjects.index[s]
        print(f"Error processing subject {failed_subject}: {str(e)}")
        failed_subjects.append(failed_subject)
        continue

# Print summary of failed subjects at the end
if failed_subjects:
    print("\nThe following subjects failed processing:")
    for subject in failed_subjects:
        print(f"- {subject}")
else:
    print("\nAll subjects processed successfully!")

# %%

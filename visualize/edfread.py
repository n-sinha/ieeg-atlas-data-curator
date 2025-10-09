#%% 
import matplotlib
matplotlib.use('qtagg')  # Set Qt backend (auto-detects Qt5/Qt6) before importing pyplot
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import numpy as np
import typer

#%%
def main(edf_file: Path = typer.Option(..., "--edf-file", "-f", help="The EDF file full path")):
    """
    Interactive EDF file viewer.
    Opens a window showing the EEG data with MNE's interactive browser.
    """
    print(f"\n{'='*60}")
    print(f"Loading EDF file: {edf_file.name}")
    print(f"{'='*60}\n")
    
    # Load the EDF file
    edf_data = mne.io.read_raw_edf(edf_file, verbose=False, infer_types=True)
    
    # Print basic info
    print(f"Channels: {len(edf_data.ch_names)}")
    print(f"Sampling rate: {edf_data.info['sfreq']} Hz")
    print(f"Duration: {edf_data.times[-1]:.2f} seconds ({edf_data.times[-1]/60:.2f} minutes)")
    print(f"\nOpening interactive viewer...")
    print("(Close the plot window or press Ctrl+C here to exit)\n")
    
    # Plot with the interactive browser
    fig = edf_data.plot(block=True)

    return edf_data
    
# %%
if __name__ == "__main__":
    # typer.run(main(edf_file=Path("data/output/ram/sub-R1010J/ses-0/ieeg/sub-R1010J_ses-0_task-FR1_acq-bipolar_ieeg.edf")))
    typer.run(main)
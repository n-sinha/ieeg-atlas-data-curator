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
    edf_data = mne.io.read_raw_edf(edf_file)
    edf_data.plot()
    input("Press Enter to continue...")
    

if __name__ == "__main__":
    typer.run(main)


#!/usr/bin/env python3
"""
Simple EDF viewer using matplotlib (no Qt dependencies).
Shows a scrollable view of EEG data.
"""
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import typer
import numpy as np

def main(edf_file: Path = typer.Option(..., "--edf-file", "-f", help="The EDF file full path")):
    """
    Simple EDF file viewer using pure matplotlib.
    """
    print(f"\n{'='*60}")
    print(f"Loading: {edf_file.name}")
    print(f"{'='*60}\n")
    
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    
    # Print info
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
    
    # Ask user what to display
    print(f"\nHow many seconds to display? (default: 30): ", end="")
    try:
        duration_input = input().strip()
        duration = float(duration_input) if duration_input else 30.0
    except:
        duration = 30.0
    
    print(f"How many channels to display? (default: 20, max: {len(raw.ch_names)}): ", end="")
    try:
        n_channels_input = input().strip()
        n_channels = int(n_channels_input) if n_channels_input else 20
        n_channels = min(n_channels, len(raw.ch_names))
    except:
        n_channels = min(20, len(raw.ch_names))
    
    print(f"\nGenerating plot ({n_channels} channels, {duration} seconds)...\n")
    
    # Get data
    n_samples = int(duration * raw.info['sfreq'])
    data, times = raw[:n_channels, :n_samples]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Calculate offsets for each channel
    data_range = np.ptp(data, axis=1).max()  # Peak-to-peak range
    offsets = np.arange(n_channels) * data_range * 1.5
    
    # Plot each channel
    for i in range(n_channels):
        ax.plot(times, data[i, :] + offsets[i], linewidth=0.5, label=raw.ch_names[i])
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    ax.set_yticks(offsets)
    ax.set_yticklabels(raw.ch_names[:n_channels], fontsize=8)
    ax.set_title(f'{edf_file.stem}\n{n_channels} channels, {duration}s duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    print("✅ Plot window opened!")
    print("Close the window to exit.\n")
    plt.show()

if __name__ == "__main__":
    typer.run(main) 
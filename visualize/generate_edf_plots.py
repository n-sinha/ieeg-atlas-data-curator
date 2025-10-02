#!/usr/bin/env python3
"""
Generate PNG plots for EDF files from a list.

This script reads EDF file paths from a text file, generates time series plots
for each file, and saves them as PNG images in organized subject subdirectories.

Usage:
    python scripts/generate_edf_plots.py --edf-list data/input/openneuro/ds006233_edf.txt
    
    or with uv:
    uv run scripts/generate_edf_plots.py --edf-list data/input/openneuro/ds006233_edf.txt
"""

import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import typer
from typing import Optional
import logging
import numpy as np

# Set up logging to track progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def generate_plot(edf_file: Path, output_dir: Path) -> bool:
    """
    Generate a time series plot for an EDF file and save as PNG.
    
    Args:
        edf_file: Path to the EDF file
        output_dir: Base directory for output files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the EDF file
        logging.info(f"Reading EDF file: {edf_file}")
        raw_data = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        
        # Extract subject and session information from the path
        # Example: data/input/openneuro/ds006233/sub-026/ses-2/ieeg/sub-026_ses-2_task-picture_ieeg.edf
        parts = edf_file.parts
        
        # Find the subject ID (starts with 'sub-')
        subject_id = None
        for part in parts:
            if part.startswith('sub-'):
                subject_id = part
                break
        
        if not subject_id:
            logging.error(f"Could not extract subject ID from path: {edf_file}")
            return False
        
        # Create output subdirectory for this subject
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename (same as input but with .png extension)
        output_filename = edf_file.stem + '.png'
        output_path = subject_output_dir / output_filename
        
        # Generate the plot using matplotlib
        logging.info(f"Generating plot for {edf_file.name}")
        
        # Get data for plotting (first 30 seconds)
        duration = 30.0  # seconds
        n_channels = min(20, len(raw_data.ch_names))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get data
        data, times = raw_data[:n_channels, :int(duration * raw_data.info['sfreq'])]
        
        # Plot each channel with offset
        offsets = np.arange(n_channels) * np.max(np.abs(data)) * 2
        for i in range(n_channels):
            ax.plot(times, data[i, :] + offsets[i], linewidth=0.5)
        
        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels')
        ax.set_yticks(offsets)
        ax.set_yticklabels(raw_data.ch_names[:n_channels])
        ax.set_title(f'{edf_file.stem}')
        ax.grid(True, alpha=0.3)
        
        # Save the figure as PNG
        logging.info(f"Saving plot to: {output_path}")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        
        logging.info(f"✅ Successfully processed: {edf_file.name}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error processing {edf_file}: {e}")
        return False


def main(
    edf_list: Path = typer.Option(
        ..., 
        "--edf-list", 
        "-l", 
        help="Path to text file containing list of EDF files"
    ),
    output_dir: Path = typer.Option(
        "outputs/ds006233_plots",
        "--output-dir",
        "-o",
        help="Base directory for output PNG files"
    )
):
    """
    Generate PNG plots for all EDF files listed in a text file.
    
    This script reads a list of EDF file paths, generates time series plots
    for each file using MNE, and saves them as PNG images organized by subject.
    """
    # Validate input file exists
    if not edf_list.exists():
        logging.error(f"EDF list file not found: {edf_list}")
        raise typer.Exit(code=1)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir.absolute()}")
    
    # Read the list of EDF files
    logging.info(f"Reading EDF file list from: {edf_list}")
    with open(edf_list, 'r') as f:
        edf_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logging.info(f"Found {len(edf_files)} EDF files to process")
    
    # Process each EDF file
    successful = 0
    failed = 0
    
    print("\n" + "="*80)
    print("STARTING EDF PLOT GENERATION")
    print("="*80 + "\n")
    
    for i, edf_path_str in enumerate(edf_files, 1):
        edf_path = Path(edf_path_str)
        
        print(f"\n[{i}/{len(edf_files)}] Processing: {edf_path.name}")
        
        # Check if file exists
        if not edf_path.exists():
            logging.warning(f"File not found: {edf_path}")
            failed += 1
            continue
        
        # Generate the plot
        if generate_plot(edf_path, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("="*80 + "\n")
    
    if failed > 0:
        logging.warning(f"Some files failed to process. Check logs above for details.")


if __name__ == "__main__":
    typer.run(main) 
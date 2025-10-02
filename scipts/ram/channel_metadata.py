#%%
import pandas as pd
import typer
from pathlib import Path
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%

def read_metadata(input_dir: Path):
    metadata_file = input_dir / "sourcedata" / "clinical_data_summary.xlsx"
    metadata = pd.read_excel(metadata_file)
    # remove patients with engel outcome more than 1
    normative_patients = metadata[metadata['engel_score'] == 1]
    # remove Cleveland Clinic: they didn't share data
    normative_patients = normative_patients[normative_patients['clinical_center'] != 'cc']
    # remove jhh: there is only one patient with many missing values
    normative_patients = normative_patients[normative_patients['clinical_center'] != 'jhh']
    return normative_patients

def main(
    input_dir: Path = typer.Option(Path("data/input/openneuro/ds003029"), "--input-dir", "-i", help="The input directory"),
    output_dir: Path = typer.Option(Path("data/output/openneuro/ds003029"), "--output-dir", "-o", help="The output directory")
):
    normative_patients = read_metadata(input_dir)
    logging.info(f"Found {len(normative_patients)} normative patients from {normative_patients['clinical_center'].unique()}")

    copy_data(input_dir, output_dir, normative_patients)

  

#%%
if __name__ == "__main__":
    typer.run(main)

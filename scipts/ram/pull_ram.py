#%%
import pandas as pd
import typer
from pathlib import Path
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%
def curate_ram(channel_metadata: pd.DataFrame):

    # remove patients that have all of these columns empty
    channel_metadata = channel_metadata[channel_metadata['Seizure Onset Zone'].notna() | 
                                        channel_metadata['Interictal Spikes'].notna() | 
                                        channel_metadata['Bad Electrodes'].notna() | 
                                        channel_metadata['Brain Lesions'].notna() | 
                                        channel_metadata['Early Spread'].notna()]

    # remove patient where seizure onset zone is empty
    channel_metadata = channel_metadata[channel_metadata['Seizure Onset Zone'].notna()]

    # make a copy of patient_id as a new column and replace everything after _ with an empty string
    channel_metadata['ram_id'] = 'sub-' + channel_metadata['patient_id'].str.split('_').str[0]

    # make clean patient id the index
    channel_metadata = channel_metadata.set_index('ram_id')

    # Add a new columun as site which will have the last letter of the ram_id
    channel_metadata['site'] = channel_metadata.index.str[-1]

    # remove patients that are from site "P" which are Penn patients
    channel_metadata = channel_metadata[channel_metadata['site'] != 'P']

    return channel_metadata

def download_ram_release(ram_input_dir: Path, dataset_id: str ):

    # create a new directory for the dataset
    dataset_dir = ram_input_dir / dataset_id

    # check if the dataset is already downloaded
    if not dataset_dir.exists():
        # clone the data from openneuro
        subprocess.run(["git", "clone", 
                        f"https://github.com/OpenNeuroDatasets/{dataset_id}.git", 
                        dataset_dir.absolute()], check=True)
    else:
        logging.info(f"Dataset {dataset_id} already downloaded")
    
    # make a list of all patients in this dataset which starts with sub-
    patients_paths = [patient for patient in dataset_dir.glob('sub-*') if patient.is_dir()]
    
    return patients_paths

def curate_ram_patients(patients_paths: list[Path], channel_metadata: pd.DataFrame):

    patients_paths_all = [patient.name for  patient in patients_paths]

    patients_with_metadata =  channel_metadata.index.unique().tolist()

    # for each patient in patients_with_metadata check if there is patient.name in patient path
    patients_paths_atlas = []
    for patient in patients_with_metadata:
        if patient in patients_paths_all:
            patients_paths_atlas.append(patients_paths[patients_paths_all.index(patient)])

    return patients_paths_atlas

def download_curated_ram(patients_path_atlas: list[Path], ram_output_dir: Path):

    for patient_path in patients_path_atlas:
        logging.info(f"Downloading patient {patient_path.name}")
        dataset_id = patient_path.parent.name + '/' + patient_path.name
        output_dir = ram_output_dir.joinpath(patient_path.name)
        subprocess.run(["aws", "s3", "sync", 
                        "--no-sign-request", 
                        f"s3://openneuro.org/{dataset_id}", 
                        output_dir.absolute()], check=True)

def main():

    ram_input_dir = Path(__file__).parent.parent.parent / "data" / "input" / "ram"
    ram_output_dir = Path(__file__).parent.parent.parent / "data" / "output" / "ram"
    channel_metadata_file = ram_output_dir / "channel_metadata.csv"
    channel_metadata = pd.read_csv(channel_metadata_file)

    channel_metadata = curate_ram(channel_metadata)

    # clone data from openneuro
    ram_datasets = pd.read_csv(ram_input_dir / "openneuro_release.csv")['dataset_id'].tolist()

    patients_paths = []

    # download the data from openneuro
    for dataset_id in ram_datasets:
        patients_paths_dataset = download_ram_release(ram_input_dir, dataset_id)
        patients_paths.extend(patients_paths_dataset) 

    # curate the patients
    patients_path_atlas = curate_ram_patients(patients_paths, channel_metadata)

    # download the data from openneuro
    download_curated_ram(patients_path_atlas, ram_output_dir)  

#%%
if __name__ == "__main__":
    typer.run(main)

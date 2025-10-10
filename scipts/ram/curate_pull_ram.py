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
def clean_ram_metadata(channel_metadata: pd.DataFrame):
    """Clean the RAM channel metadata
    Args:
        channel_metadata: pandas DataFrame containing the channel metadata

    Returns:
        pandas DataFrame containing the curated channel metadata
    """

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

    logging.info(f"Removing patients that are from 'Penn' in RAM dataset")
    channel_metadata = channel_metadata[channel_metadata['site'] != 'P']

    return channel_metadata

def git_clone(directory: Path, dataset_id: str ):

    # create a new directory for the dataset
    dataset_dir = directory / dataset_id


    # clone the data from openneuro
    if dataset_dir.exists():
        # Pull latest changes if directory exists
        logging.info(f"Pulling latest changes for {dataset_id}")
        subprocess.run(["git", "-C", dataset_dir.absolute(), "pull"], check=True)
    else:
        # Clone if directory doesn't exist
        logging.info(f"Cloning {dataset_id}")
        subprocess.run(["git", "clone", 
                    f"https://github.com/OpenNeuroDatasets/{dataset_id}.git", 
                    dataset_dir.absolute()], check=True)
    
    # make a list of all patients in this dataset which starts with sub-
    patients_paths = [patient for patient in dataset_dir.glob('sub-*') if patient.is_dir()]
    
    return patients_paths

def curate_ram_patients(patients_paths: list[Path], channel_metadata: pd.DataFrame):

    patients_paths_all = [patient.name for  patient in patients_paths]
    logging.info(f"Found {len(patients_paths_all)} patients across all ram datasets")

    patients_with_metadata =  channel_metadata.index.unique().tolist()
    logging.info(f"{len(patients_with_metadata)} patients have channel level metadata")

    # for each patient in patients_with_metadata check if there is patient.name in patient path
    patients_paths_atlas = []
    for patient in patients_with_metadata:
        if patient in patients_paths_all:
            patients_paths_atlas.append(patients_paths[patients_paths_all.index(patient)])
    logging.info(f"{len(patients_paths_atlas)} patients with channel metadata in assests are in the ram datasets")

    return patients_paths_atlas

def pull_ram_data(patients_path_atlas: list[Path], ram_output_dir: Path, num_patients: int = None):

    if num_patients is not None:
        patients_path_atlas = patients_path_atlas[:num_patients]
        logging.info(f"Downloading {num_patients} patients")
    else:
        logging.info(f"Downloading all {len(patients_path_atlas)} patients locally")

    for patient_path in patients_path_atlas:
        logging.info(f"Downloading patient {patient_path.name}")
        dataset_id = patient_path.parent.name + '/' + patient_path.name
        output_dir = ram_output_dir.joinpath(patient_path.name)
        subprocess.run(["aws", "s3", "sync", 
                        "--no-sign-request", 
                        f"s3://openneuro.org/{dataset_id}", 
                        output_dir.absolute()], check=True)

def main(n_patients: int = None):

    project_root = Path(__file__).parent.parent.parent

    ram_input_dir = project_root / "data" / "input" / "ram"
    ram_output_dir = project_root / "data" / "output" / "ram"
    channel_metadata_file = ram_input_dir / "channel_metadata.csv"
    channel_metadata = pd.read_csv(channel_metadata_file)

    channel_metadata = clean_ram_metadata(channel_metadata)

    # clone data from openneuro
    ram_datasets = pd.read_csv(project_root / 
                               "assets" / 
                               "ram_metadata" / 
                               "openneuro_release_ram.csv")['dataset_id'].tolist()

    patients_paths = []

    # download the data from openneuro
    for dataset_id in ram_datasets:
        patients_paths_dataset = git_clone(directory=ram_input_dir, dataset_id=dataset_id)
        patients_paths.extend(patients_paths_dataset)

    # curate the patients
    patients_path_atlas = curate_ram_patients(patients_paths, channel_metadata)

    # download the data from openneuro
    pull_ram_data(patients_path_atlas, ram_output_dir, num_patients=n_patients)  

#%%
if __name__ == "__main__":
    typer.run(main)

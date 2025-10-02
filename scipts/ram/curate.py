#%%
import pandas as pd
import typer
from pathlib import Path
import logging
import shutil
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#%%

def text_to_dict(input_dir: Path):
    logging.info(f"Processing {input_dir}")
    
    # Find the file
    if input_dir.is_file():
        file_path = input_dir
    else:
        txt_files = list(input_dir.glob("*electrode_categories*.txt"))
        if not txt_files:
            return pd.DataFrame()
        file_path = txt_files[0]
    
    # Read and parse
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    patient_id = lines[0]
    electrode_categories = {}
    current_category = None
    
    for line in lines[1:]:
        # Category headers have spaces, colons, or are long
        if ' ' in line or ':' in line or len(line) > 10:
            current_category = line
            electrode_categories[current_category] = []
        # Filter out non-channel descriptors (like 'interictal' which means interictal spike)
        elif current_category and line.lower() not in ['-', 'none', 'n/a', 'interictal']:
            electrode_categories[current_category].append(line)

    # get keys and values from electrode_categories
    keys = electrode_categories.keys()
    values = electrode_categories.values()
    return keys, values, patient_id

def main():

    data_dir = Path(__file__).parent.parent.parent / "data" / "input" / "ram"
    release_2016 = data_dir / "Release_Metadata_20160930" / "electrode_categories"
    release_2017 = data_dir / "Release_Metadata_20171010" / "electrode_categories"
    release_2018 = data_dir / "Release_Metadata_20180528" / "electrode_categories"

    subjects_2016 = release_2016.glob("*")
    subjects_2017 = release_2017.glob("*")
    subjects_2018 = release_2018.glob("*")

    # keep only the files that start with R
    subjects_2016 = [subject for subject in subjects_2016 if subject.name.startswith("R")]
    subjects_2017 = [subject for subject in subjects_2017 if subject.name.startswith("R")]
    subjects_2018 = [subject for subject in subjects_2018 if subject.name.startswith("R")]

    output_dir = data_dir.parent.parent / "output" / "ram"
    if output_dir.exists():
        shutil.rmtree(output_dir)  # Remove directory and all contents
    output_dir.mkdir(parents=True)  # Create empty directory

    for subject in list(subjects_2016) + list(subjects_2017) + list(subjects_2018):
        keys, values, patient_id = text_to_dict(subject)
        # create a dictionary
        dictionary = {key: value for key, value in zip(keys, values)}
        with open(output_dir / f"sub-{patient_id}.json", "w") as f:
            json.dump(dictionary, f, indent=4)
  

#%%
if __name__ == "__main__":
    typer.run(main)

# BIDS Individual Subject File Generator

This script generates BIDS-compliant files for each subject in the output BIDS directory. For each subject, it creates three files in the `primary` directory:

1. **dataset_description.json** - BIDS 1.10 compliant with modifications
2. **participants.json** - JSON schema for participant metadata  
3. **participants.tsv** - Tab-separated participant data (participant_id, age, sex)

## Features

- **Automatic Subject Matching**: Matches subjects from the output BIDS directory with participants from multiple input datasets (ds004789, ds004809, ds004865, ds005059, ds005411)
- **BIDS 1.10 Compliance**: Generates files that comply with the latest BIDS specification
- **Reproducible**: Creates consistent, reproducible BIDS files for each subject
- **Comprehensive Metadata**: Includes original dataset information, funding details, and authorship

## Usage

### Basic Usage
```bash
python scipts/ram/generate_bids_files.py
```

### What the Script Does

1. **Loads Participant Data**: Reads `participants.tsv` and `dataset_description.json` files from all input datasets
2. **Matches Subjects**: Finds participant information for each subject in the output BIDS directory
3. **Generates Files**: Creates the three required BIDS files for each subject in their `primary` directory

### File Structure Generated

For each subject (e.g., `sub-R1010J`), the script creates:

```
/users/nishants/ieeg-atlas-data-curator/data/output/ram/BIDS/sub-R1010J/primary/
├── dataset_description.json
├── participants.json
└── participants.tsv
```

### Example Output

**dataset_description.json**:
```json
{
    "Name": "Individual Subject Dataset: sub-R1010J",
    "BIDSVersion": "1.10.0",
    "DatasetType": "raw",
    "Authors": [
        "Haydn G. Herrema",
        "Michael J. Kahana", 
        "Nishant Sinha"
    ],
    "Funding": [
        "NINDS K99NS138680 (PI: Nishant Sinha)"
    ],
    "License": "CC0",
    "GeneratedBy": [...],
    "SourceDatasets": [...]
}
```

**participants.tsv**:
```
participant_id	age	sex
sub-R1010J	30	F
```

## Requirements

- Python 3.6+
- pandas
- pathlib (built-in)
- json (built-in)

## Author

**Nishant Sinha**  
Email: Nishant.Sinha@Pennmedicine.upenn.edu

## License

This script is part of the iEEG Atlas Data Curator project and follows the same licensing terms as the main project.

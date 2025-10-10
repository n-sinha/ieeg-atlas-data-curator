import pandas as pd
from pathlib import Path
import re

data_dir = Path(__file__).parent.parent.parent / "assets" / "ram_metadata"
input_dirs = [
    data_dir / "Release_Metadata_20160930" / "electrode_categories",
    data_dir / "Release_Metadata_20171010" / "electrode_categories",
    data_dir / "Release_Metadata_20180528" / "electrode_categories",
]
output_file = Path(__file__).parent.parent.parent / "data" / "input" / "ram" / "channel_metadata.csv"

# Known valid category names (normalized)
VALID_CATEGORIES = {
    'seizure onset zone': 'Seizure Onset Zone',
    'seizure onset zones': 'Seizure Onset Zone',
    'seizure onset': 'Seizure Onset Zone',
    'ictal onset': 'Seizure Onset Zone',
    'octal onset': 'Seizure Onset Zone',
    'octal onset zone': 'Seizure Onset Zone',
    'interictal spikes': 'Interictal Spikes',
    'interictal spiking': 'Interictal Spikes',
    'significant interictal spiking': 'Interictal Spikes',
    'significant interitcal spiking': 'Interictal Spikes',
    'bad electrodes': 'Bad Electrodes',
    'bad electrode': 'Bad Electrodes',
    'bad channels': 'Bad Electrodes',
    'broken leads': 'Bad Electrodes',
    'broken lead': 'Bad Electrodes',
    'brain lesions': 'Brain Lesions',
    'brain lesion': 'Brain Lesions',
    'early spread': 'Early Spread',
}

def is_valid_patient_id(text):
    return bool(re.match(r'^R\d{4}[A-Z](_\d)?$', text))

def is_note_separator(line):
    return line.startswith('***') or line.startswith('**') or line.startswith('--')

def normalize_category(category):
    category_lower = category.lower().strip().rstrip(':').strip()
    return VALID_CATEGORIES.get(category_lower, None)

txt_files = []
for input_dir in input_dirs:
    if input_dir.exists():
        txt_files.extend([f for f in input_dir.glob("*.txt") if not f.name.startswith('.')])

rows = []
for file_path in sorted(txt_files):
    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) == 0:
        continue
    
    patient_id = lines[0]
    
    if not is_valid_patient_id(patient_id):
        continue
    
    row = {'patient_id': patient_id}
    current_category = None
    
    for line in lines[1:]:
        line = line.rstrip('%').strip()
        
        if is_note_separator(line):
            break
        
        normalized = normalize_category(line)
        if normalized:
            current_category = normalized
            if current_category not in row:
                row[current_category] = []
        elif current_category and line.lower() not in ['-', 'none', 'n/a', 'interictal', '']:
            if re.match(r'^[A-Za-z0-9_/\-]+\d+[A-Za-z0-9_/\-]*$', line, re.IGNORECASE):
                row[current_category].append(line)
    
    for key in row:
        if isinstance(row[key], list):
            row[key] = ', '.join(row[key]) if row[key] else ''
    
    rows.append(row)

df = pd.DataFrame(rows)
df = df.fillna('')

column_order = ['patient_id', 'Seizure Onset Zone', 'Interictal Spikes', 'Bad Electrodes', 'Brain Lesions', 'Early Spread']
existing_columns = [col for col in column_order if col in df.columns]
df = df[existing_columns]

df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
print(f"Shape: {df.shape}")
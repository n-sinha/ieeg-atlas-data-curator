import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

data_dir = Path(__file__).parent.parent.parent / "data"
input_dirs = [
    data_dir / "input" / "ram" / "Release_Metadata_20160930" / "electrode_categories",
    data_dir / "input" / "ram" / "Release_Metadata_20171010" / "electrode_categories",
    data_dir / "input" / "ram" / "Release_Metadata_20180528" / "electrode_categories",
]
output_file = data_dir / "output" / "ram" / "channel_metadata_audit.txt"

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

def is_electrode_name(line):
    return bool(re.match(r'^[A-Za-z0-9_/\-]+\d+[A-Za-z0-9_/\-]*$', line, re.IGNORECASE))

def normalize_category(category):
    category_lower = category.lower().strip().rstrip(':').strip()
    return category_lower

mapped_categories = defaultdict(int)
ignored_categories = defaultdict(int)
all_raw_categories = defaultdict(int)

txt_files = []
for input_dir in input_dirs:
    if input_dir.exists():
        txt_files.extend([f for f in input_dir.glob("*.txt") if not f.name.startswith('.')])

for file_path in sorted(txt_files):
    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) == 0:
        continue
    
    patient_id = lines[0]
    
    if not is_valid_patient_id(patient_id):
        continue
    
    for line in lines[1:]:
        line = line.rstrip('%').strip()
        
        if is_note_separator(line):
            break
        
        if line.lower() in ['-', 'none', 'n/a', 'interictal', '']:
            continue
        
        if is_electrode_name(line):
            continue
        
        normalized = normalize_category(line)
        all_raw_categories[line] += 1
        
        if normalized in VALID_CATEGORIES:
            mapped_target = VALID_CATEGORIES[normalized]
            mapped_categories[f"{line} -> {mapped_target}"] += 1
        else:
            ignored_categories[line] += 1

with open(output_file, "w") as f:
    f.write("="*80 + "\n")
    f.write("CATEGORY AUDIT REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total text files processed: {len(txt_files)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("MAPPED CATEGORIES (included in output)\n")
    f.write("="*80 + "\n\n")
    for category, count in sorted(mapped_categories.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{category:<60} Count: {count:>4}\n")
    
    f.write("\n\n")
    f.write("="*80 + "\n")
    f.write("IGNORED CATEGORIES (not included in output)\n")
    f.write("="*80 + "\n\n")
    if ignored_categories:
        for category, count in sorted(ignored_categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{category:<60} Count: {count:>4}\n")
    else:
        f.write("None - all categories were mapped!\n")
    
    f.write("\n\n")
    f.write("="*80 + "\n")
    f.write("MAPPING RULES\n")
    f.write("="*80 + "\n\n")
    for original, mapped in sorted(VALID_CATEGORIES.items()):
        f.write(f"{original:<40} -> {mapped}\n")

print(f"Audit report saved to: {output_file}")
print(f"\nSummary:")
print(f"  Mapped categories: {len(mapped_categories)}")
print(f"  Ignored categories: {len(ignored_categories)}") 
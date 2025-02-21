# Find Interictal Clips Script Explanation

## 1. Main Purpose
The purpose of this script is to identify optimal time periods for collecting interictal (non-seizure) brain activity data during awake states, while ensuring that these periods are distant from seizure events and other confounding factors.

---

## 2. Workflow Overview

### 2.1 Setup and Data Loading
- **Load configuration and paths.**
- **Load metadata:**
  - Seizure metadata (from Excel)
  - Sleep times data (from a pickle file)
- **Identify list of patients to analyze.**

### 2.2 Set Constants
```python
PREICTAL_WINDOW = 30  # seconds
IMPLANT_BUFFER = 72 * 3600  # 72 hours in seconds
SEIZURE_BUFFER = 2 * 3600   # 2 hours in seconds
BEFORE_BUFFER = 2 * 3600    # 2 hours in seconds
```

### 2.3 Pseudocode for Each Patient
```python
for each patient:
    # Step 1: Create masks for valid time periods
    wake_mask = find times when patient is awake
    outside_seizure_mask = find times that are not near seizure events
    implant_mask = find times after the implant effect period

    # Step 2: Combine the masks to determine valid periods
    valid_periods = wake_mask AND outside_seizure_mask AND implant_mask

    # Step 3: Identify continuous segments within the valid periods
    segments = find_continuous_segments(valid_periods)
    segments = sort_by_length(segments)  # longest segments first

    # Step 4: Randomly select clip times from the top segments
    for each of the top 2 longest segments:
        randomly select 10 non-overlapping clips of 30 seconds each
        save clip times and corresponding file numbers
```

---

## 3. Visualizations

### Timeline Visualization
```
─────────────────────────────────────────────────────────────
│ Implant │    Recording Period                             │
─────────────────────────────────────────────────────────────
     ▲          ▲       ▲        ▲       ▲
     │          │       │        │       │
   Implant    Seizure  Sleep   Seizure  Sleep
   Effect      Event   Period   Event   Period
```

### Valid Periods Example
```
─────────────────────────────────────────────────────────────
│ ❌ │ ✅ │ ❌ │ ❌ │ ✅ │ ❌ │ ❌ │ ✅ │ ❌ │
─────────────────────────────────────────────────────────────
```

### Selected Clips (30-second segments)
```
─────────────────────────────────────────────────────────────
│    │ *│* │    │    │ *│* │    │    │ *│* │    │
─────────────────────────────────────────────────────────────
        ↑   ↑              ↑   ↑              ↑   ↑
      Random 30s         Random 30s          Random 30s
       Clips              Clips               Clips 
```

---

## 4. Key Points Recap

1. **Criteria for Valid Time Periods:**
   - Occur during wake time.
   - Are buffered away from seizures.
   - Occur after the implant effect period.
   - Lie within the same recording file.

2. **Process Summary:**
   - Identify the longest continuous valid segments.
   - Randomly select 10 non-overlapping clips from each of the top two segments.

3. **Outputs Generated:**
   - `best_interictal_times.npy`: Array of best interictal periods (start and end times).
   - `best_interictal_times_filenum.npy`: File numbers corresponding to each period.
   - `interictal_clip_times.npy`: Selected clip start times.
   - `interictal_clip_files.npy`: File numbers for each clip.

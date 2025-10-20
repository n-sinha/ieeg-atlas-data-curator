# Turnkey OpenNeuro to BIDS Pipeline - Context Save

## Current Status (October 19, 2025)

### ✅ Completed Components

1. **Turnkey Solution Structure**: Created in `scipts/turnkey/`
   - `openneuro_to_bids.py`: Main orchestrator script
   - `test_pipeline.py`: Test script for sub-R1010J
   - `README.md`: Comprehensive documentation
   - `Makefile`: Easy execution commands
   - `config.py`: Configuration settings

2. **Pipeline Integration**: Successfully integrated all existing scripts:
   - `channel_metadata.py`: Channel metadata creation ✅
   - `curate_pull_ram.py`: OpenNeuro data pulling ✅
   - `standardize_ram.py`: RAM data standardization ✅
   - `standardize_channels.py`: Channel metadata derivatives (in progress)
   - `reorganize_ieeg_clips.py`: BIDS reorganization
   - `deploy_dataset_description.py`: Dataset description deployment

3. **UV Integration**: Updated all scripts to use `uv run` for dependency management

### 🔄 Current Progress

**Pipeline Steps Completed:**
1. ✅ **Step 1**: Channel metadata creation from assets
2. ✅ **Step 2**: Data pulling from OpenNeuro (pulled sub-R1310J)
3. ✅ **Step 3**: RAM data standardization (took ~1 minute)
4. 🔄 **Step 4**: Channel metadata derivative creation (needs fix)
5. ⏳ **Step 5**: BIDS compliance (pending)

### 🐛 Current Issue

**Problem**: `standardize_channels.py` script doesn't accept command line arguments
- Script processes all subjects in `data/output/ram/BIDS/` automatically
- Need to modify the turnkey solution to work with this behavior

**Error**: 
```
Failed to create channel metadata derivative: Usage: standardize_channels.py [OPTIONS]
No such option: --subject-dir
```

### 📁 File Structure Created

```
scipts/turnkey/
├── openneuro_to_bids.py      # Main orchestrator (uses uv run)
├── test_pipeline.py          # Test script
├── README.md                 # Documentation
├── Makefile                  # make test, make run, make clean
└── config.py                 # Configuration settings
```

### 🔧 Technical Details

**Path Resolution Fixed:**
- Project root: `/mnt/sauce/littlab/users/nishants/ieeg-atlas-data-curator`
- RAM directory: `scipts/ram/`
- Uses `uv run` for all subprocess calls

**Data Flow:**
1. `uv run channel_metadata.py` → Creates `data/input/ram/channel_metadata.csv`
2. `uv run curate_pull_ram.py --n-patients 1` → Downloads subject data
3. `uv run standardize_ram.py` → Processes and creates BIDS structure
4. `uv run standardize_channels.py` → Creates channel metadata derivatives
5. `uv run reorganize_ieeg_clips.py` → Makes BIDS compliant
6. `uv run deploy_dataset_description.py` → Deploys dataset descriptions

### 🎯 Next Steps

1. **Fix Step 4**: Modify `create_channel_metadata_derivative()` function to work with `standardize_channels.py`'s automatic processing
2. **Complete Pipeline**: Finish Steps 4 and 5
3. **Test Full Pipeline**: Run complete test with sub-R1010J
4. **Documentation**: Update README with final usage instructions

### 💡 Key Insights

- **UV Integration**: Successfully resolved dependency issues by using `uv run`
- **Path Management**: Fixed project root calculation for correct file paths
- **Subprocess Approach**: Using subprocess calls instead of direct imports avoids dependency conflicts
- **RAM Script Behavior**: `curate_pull_ram.py` downloads all patients, `standardize_channels.py` processes all subjects automatically

### 🚀 Usage Commands

```bash
# Test the pipeline
cd scipts/turnkey
uv run python test_pipeline.py

# Or use make
make test

# Manual execution
uv run python openneuro_to_bids.py --dataset-id ds003505 --subject-id sub-R1010J
```

### 📊 Current Test Results

- **Channel Metadata**: ✅ Created successfully
- **Data Pulling**: ✅ Downloaded sub-R1310J (not sub-R1010J as expected)
- **Standardization**: ✅ Completed in ~1 minute
- **Channel Derivatives**: ❌ Failed due to argument mismatch
- **BIDS Compliance**: ⏳ Pending

### 🔍 Files to Check

- `turnkey_pipeline.log`: Detailed execution log
- `data/output/ram/BIDS/sub-R1310J/`: Generated BIDS structure
- `data/input/ram/channel_metadata.csv`: Channel metadata file

---

**Status**: 80% Complete - Ready to fix final integration issues and complete testing



# %%
from glob import glob
import os
import sys
from unicodedata import name

code_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(code_path)

import warnings
import json
from os.path import join as ospj
import numpy as np
import pandas as pd
import tools
from pqdm.processes import pqdm
import re

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"  # for parallel
warnings.filterwarnings("ignore")

# %% [markdown]
# ## set Params

# %%
with open(ospj(code_path, "config.json"), "rb") as f:
    config = json.load(f)
repo_path = config["repositoryPath"]
metadata_path = ospj(repo_path, "ieeg-metadata")
data_path = ospj(repo_path, "data")

# from erin's sheet below
seizure_metadata = pd.read_excel(
    ospj(metadata_path, "Manual validation.xlsx"), sheet_name="AllSeizureTimes"
).dropna(how="all")

# %%
patients = pd.read_csv(ospj(metadata_path, "patient_list.csv"))["patient"]


# %%
def LoopPatientBandpower(patient):
    files = glob(
        ospj(data_path, patient, time_point, f"{time_point}_eeg_bipolar_*.pkl")
    )
    for f in files:
        # here do some filtering for CAR vs Bipolar reference
        f_num = re.search(r"(\d+)\.pkl$", f).group(1)
        tools.SaveBandpowerFromData(
            data_path,
            patient,
            time_point,
            f_num,
        )
        tools.SaveNetworksFromData(
            data_path,
            patient,
            time_point,
            f_num,
        )


def TryLoop(patient):
    try:
        LoopPatientBandpower(patient)
    except:
        pass


def Initialize(t, d):
    global time_point
    time_point = t
    global data_path
    data_path = d


# %%
if __name__ == "__main__":
    for time_point in ["interictal"]:
        pqdm(
            patients,
            TryLoop,
            n_jobs=20,
            initializer=Initialize,
            initargs=(time_point, data_path),
        )

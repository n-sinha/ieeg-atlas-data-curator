# LoadFeatNets.py

from logging import exception
from os.path import join as ospj
from os.path import exists
from glob import glob
import numpy as np
import pandas as pd
import re


def LoadNetworks(data_path, patient, time_periods):
    PtDf = []
    for period in time_periods:
        if not exists(ospj(data_path, patient, period)):
            continue
        nx_files = glob(
            ospj(data_path, patient, period, f"{period}_networks_bipolar_*.pkl")
        )
        length = len(nx_files)
        for f in nx_files:
            f_num = int(re.search(r"(\d+)\.pkl$", f).group(1))
            data = (
                pd.read_pickle(f)
                .assign(patient=patient, period=period, clip=f_num)
                .set_index(["patient", "period", "clip"], append=True)
                .reorder_levels(["patient", "period", "clip", "channel_1", "channel_2"])
            )
            PtDf.append(data.copy())
            ## needs some tracking of patient/period/clip
    PtDf = pd.concat(PtDf)
    return PtDf

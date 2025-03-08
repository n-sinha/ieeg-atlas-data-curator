# LoadFeatNets.py

from os.path import exists
from os.path import join as ospj
from glob import glob
import numpy as np
import pandas as pd
import re


def LoadBandpower(data_path, patient, time_periods):
    if not (
        np.all([exists(ospj(data_path, patient, period)) for period in time_periods])
    ):
        raise
    PtDf = []
    for period in time_periods:
        files = glob(
            ospj(data_path, patient, period, f"{period}_bandpower_bipolar*.pkl")
        )
        for f in files:
            f_num = int(re.search(r"(\d+)\.pkl$", f).group(1))
            # f_num = int(re.search(r"(\d+)\.(\w+)$", f).group(1))
            # here do some filtering for CAR vs Bipolar reference
            data = (
                pd.read_pickle(f)
                .rename_axis(index="freq")
                .assign(patient=patient, period=period, clip=f_num)
                .set_index(["patient", "period", "clip"], append=True)
            )
            # data = data.divide(data.sum())
            PtDf.append(data.copy())
    PtDf = pd.concat(PtDf)
    PtDf = (
        PtDf.melt(ignore_index=False, var_name="channel", value_name="bandpower")
        .set_index("channel", append=True)
        .reorder_levels(["patient", "period", "clip", "channel", "freq"])
    )
    return PtDf

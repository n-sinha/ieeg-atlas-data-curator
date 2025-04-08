from os.path import join as ospj
import numpy as np
import pandas as pd
from .get_network_coherence_from_data import get_network_coherence_from_data


def SaveNetworksFromData(data_path, patient, time_point, f_num):

    f = ospj(data_path, patient, time_point, f"{time_point}_eeg_bipolar_{f_num}.pkl")

    data = pd.read_pickle(f)

    # get network
    nxx = get_network_coherence_from_data(existing_data=data)

    nxx.to_pickle(
        ospj(
            data_path,
            patient,
            time_point,
            f"{time_point}_networks_bipolar_{f_num}.pkl",
        )
    )

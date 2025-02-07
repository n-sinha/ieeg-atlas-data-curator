from os.path import join as ospj
import numpy as np
import pandas as pd
from tools.get_atlas_bandpower_from_data import get_atlas_bandpower_from_data


def SaveBandpowerFromData(data_path, patient, time_point, f_num):

    f = ospj(data_path, patient, time_point, f"{time_point}_eeg_bipolar_{f_num}.pkl")

    data = pd.read_pickle(f)

    # get network
    pxx = get_atlas_bandpower_from_data(existing_data=data)

    pxx.to_pickle(
        ospj(
            data_path,
            patient,
            time_point,
            f"{time_point}_bandpower_bipolar_{f_num}.pkl",
        )
    )
    # np.save(ospj(data_path, patient, f"interictal_networks_{f_num}.npy"), network_bands)

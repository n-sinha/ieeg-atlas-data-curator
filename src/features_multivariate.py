#%%
import pandas as pd
import numpy as np
import h5py
from process_ieeg import IEEGClipProcessor
from IPython import embed
from pathlib import Path

#%%
class MultivariateFeatures(IEEGClipProcessor):
    def __init__(self, subject_id: str):
        super().__init__()
        self.subject_id = subject_id
        self.ieeg_processed  = next(self.project_root.joinpath('data', 'derivatives').rglob(f'{subject_id}/**/interictal_ieeg_processed.h5'))
        self.ieeg_processed_bipolar, self.coordinates, self.sampling_frequency = self.load_ieeg()

    def load_ieeg(self):
        if not self.ieeg_processed.exists():
            raise FileNotFoundError(f"No iEEG processed file found for subject {self.subject_id}")
        
        with h5py.File(self.ieeg_processed, 'r') as f:
                coordinates_data = f['/bipolar_montage/coordinates']
                ieeg_data = f['/bipolar_montage/ieeg']
                ieeg = pd.DataFrame(ieeg_data, columns=ieeg_data.attrs['channels_labels'])
                sampling_frequency = ieeg_data.attrs['sampling_rate']
                coordinates = pd.concat([
                    pd.DataFrame(coordinates_data, columns=['x', 'y', 'z'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['original_labels'], columns=['orig_labels'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roi'], columns=['roi'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['roiNum'], columns=['roiNum'], index=coordinates_data.attrs['labels']),
                    pd.DataFrame(coordinates_data.attrs['spared'], columns=['spared'], index=coordinates_data.attrs['labels'])
                ], axis=1)
        
        return ieeg, coordinates, sampling_frequency
    
    # add multivariate features here
    
    def features_1(self):
        pass

    def features_2(self):
        pass

    def features_3(self):
        pass


#%%
if __name__ == "__main__":
    subject_id = "sub-RID0031"
    features = MultivariateFeatures(subject_id)
    print(features.ieeg_processed_bipolar.head())
    print(features.coordinates.head())


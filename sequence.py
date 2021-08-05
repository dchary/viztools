import numpy as np
from typing import Union
from copy import copy
import viztools as vz
"""
A simple object to hold the image data
"""
class WellSequence():

    def __init__(self, X):
        self._X = X

        self.shape = self._X.shape[2:]
        self.timesteps = self._X.shape[0]
        self.channels = self._X.shape[1]

        self._channels = {'Bystander' : 0,
                'Ontarget' : 1,
                'Death' : 2,
                'T Cell' : 3,
                'Brightfield' : 4}

        self.uns = dict()

    def set_X(self, X : np.array):
        self._X = X
        self._update_metadata()
    
    def get_X(self):
        return self._X

    def get_channel(self, channel : Union[str, int]):
        if isinstance(channel, str):
            return self._X[:, self._channels.get(channel),:,:]
        elif isinstance(channel, int):
            return self._X[:, channel, :, :]
        else:
            raise NotImplemented
    
    def get_channel_timestep(self, channel : Union[str, int] = 'Brightfield', timesteps : Union[int, tuple] = 0):
        X_channel = self.get_channel(channel)
        if isinstance(timesteps, int):
            return X_channel[timesteps,:,:]
        elif isinstance(timesteps, tuple) or (isinstance(timesteps, list) & len(timesteps) == 2):
            return X_channel[timesteps[0]:timesteps[1], :,:]
        else:
            raise NotImplemented

    def get_timesteps(self, timesteps : Union[int, tuple] = 0):
        if isinstance(timesteps, int):
            return self._X[timesteps,:,:,:]
        elif isinstance(timesteps, tuple) or (isinstance(timesteps, list) & len(timesteps) == 2):
            return self._X[timesteps[0]:timesteps[1], :,:,:]
        else:
            raise NotImplemented

    def get_nanowell(self, i : int):
        coords = self.uns['nanowells'][i]
        nw = copy(self)
        nw.set_X(nw.get_X()[:, :, coords[1] : coords[1] + coords[2], coords[0] : coords[0] + coords[3]])
        nw.uns['nw_crop_coords'] = coords
        nw.uns['nw_idx'] = i
        return nw

    def _update_metadata(self):
        self.shape = self._X.shape[2:]
        self.timesteps = self._X.shape[0]
        self.channels = self._X.shape[1]

    def plot(self, *args, **kwargs):
        return vz.pl.plot_frame(self, *args, **kwargs)

    def mp4(self, *args, **kwargs):
        return vz.pl.to_mp4(self, *args, **kwargs)

    
"""Implements data loading from Momentum Microscope in the Baldini Lab"""
import numpy as np
import xarray as xr
import re

from arpes.endstations import HemisphericalEndstation

__all__ = ("MMEndstation",)


class MMEndstation(HemisphericalEndstation):
    """Implements loading .xy text file format for Momentum Microscope"""

    PRINCIPAL_NAME = "MM"
    ALIASES = [
        "MM_XY",
        "XY"
    ]
    _TOLERATED_EXTENSIONS = {
        ".xy", 
    }

    def print_m(self, *messages):
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.PRINCIPAL_NAME)
        print(s, *messages)
    
    def resolve_frame_locations(self, scan_desc: dict = None):
        """There is only a single h5 file for SLS data without Deflector mode, so this is simple."""
        return [scan_desc.get("path", scan_desc.get("file"))]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """
        Loads a single spectrum from .xy MM data. Mostly just a wrapper around
        functions in momentum functions, but allows for data to be loaded by
        using 'load_data(file_path, location='MM').
        """
        from momentum_functions import load_xy_data
        
        return load_xy_data(frame_path)

    def postprocess(self, frame: xr.Dataset):
        return frame

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        """
        1. Converts kinetic energy to binding energy if necessary
        """

        #Binding energy conversion
        if 'eV' in data.coords:
            workfunction = data.attrs['Eff.Workfunction:']
            if "KineticEnergy" in data.attrs['ScanVariable:']:
                photon_energy = data.attrs['ExcitationEnergy:']
                data.coords['eV'] = data.eV + np.full_like(data.eV, float(workfunction) - float(photon_energy)) 
        
        return data


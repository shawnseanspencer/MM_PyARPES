"""
Additional functions for the new capabilities of the momentum microscope. Also a few that add general functionality to PyARPES
"""

from arpes.xarray_extensions import ARPESAccessorBase, ARPESDataArrayAccessor
import xarray as xr
import numpy as np

class RealSpaceAccessor(ARPESDataArrayAccessor):
    """Class for functions extending PyARPES functionality to real space data.
    Only real space specific functions or their dependents should be written here"""
    
    def fat_sel_extended(self, widths = None, **kwargs) -> xr.DataArray:
        """Allows integrating a selection over a small region.

        The produced dataset will be normalized by dividing by the number
        of slices integrated over.

        This can be used to produce temporary datasets that have reduced
        uncorrelated noise.

        Args:
            widths: Override the widths for the slices. Resonable defaults are used otherwise. Defaults to None.
            kwargs: slice dict. Has the same function as xarray.DataArray.sel

        Returns:
            The data after selection.
        """
        if widths is None:
            widths = {}

        default_widths = {
            "eV": 0.05,
            "phi": 2,
            "beta": 2,
            "theta": 2,
            "kx": 0.02,
            "ky": 0.02,
            "kp": 0.02,
            "kz": 0.1,
            #Real space
            "x": 0.1,
            "y": 0.1
        }

        extra_kwargs = {k: v for k, v in kwargs.items() if k not in self._obj.dims}
        slice_kwargs = {k: v for k, v in kwargs.items() if k not in extra_kwargs}
        slice_widths = {
            k: widths.get(k, extra_kwargs.get(k + "_width", default_widths.get(k)))
            for k in slice_kwargs
        }

        slices = {
            k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2)
            for k, v in slice_kwargs.items()
        }

        sliced = self._obj.sel(**slices)
        thickness = np.product([len(sliced.coords[k]) for k in slice_kwargs.keys()])
        normalized = sliced.sum(slices.keys(), keep_attrs=True) / thickness
        for k, v in slices.items():
            normalized.coords[k] = (v.start + v.stop) / 2
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized

    @property
    def real_space(self):
        return self.fat_sel_extended(eV=0)
    
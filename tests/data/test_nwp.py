""" Tests for the nwp regridding module """
import os
import tempfile

import xarray as xr

from india_forecast_app.data.nwp import regrid_nwp_data


def test_regrid_nwp_data(nwp_mo_global_data):
    """Test the regridding of the nwp data"""

    # create a temporary dir
    with tempfile.TemporaryDirectory() as temp_dir:

        # save mo data to zarr
        nwp_zarr = os.environ["NWP_MO_GLOBAL_ZARR_PATH"]

        # regrid the data
        nwp_xr = xr.open_zarr(nwp_zarr)
        nwp_xr_regridded = regrid_nwp_data(
            nwp_xr, "india_forecast_app/data/mo_global/india_coords.nc"
        )

        # check the data is different in latitude and longitude
        assert not nwp_xr_regridded.latitude.equals(nwp_xr.latitude)
        assert not nwp_xr_regridded.longitude.equals(nwp_xr.longitude)

        assert len(nwp_xr_regridded.latitude) == 225
        assert len(nwp_xr_regridded.longitude) == 150

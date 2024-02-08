import xarray as xr


def run():
    ds = xr.open_dataset("data/wind/wind_data.nc")
    # ds = xr.open_dataset("tests/test_data/wind/wind_data.nc")
    print(ds["time_utc"])

    ds = xr.open_zarr("data/nwp.zarr")
    # ds = xr.open_zarr("tests/test_data/nwp.zarr")
    print(ds['init_time'])
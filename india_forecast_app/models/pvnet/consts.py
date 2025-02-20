"""Constants for PVNet data paths"""

root_data_path = "data"
nwp_path = f"{root_data_path}/nwp.zarr"
nwp_ecmwf_path = f"{root_data_path}/nwp_ecmwf.zarr"
nwp_mo_global_path = f"{root_data_path}/nwp_mo_global.zarr"
nwp_gfs_path = f"{root_data_path}/nwp_gfs.zarr"
wind_path = f"{root_data_path}/wind"
pv_path = f"{root_data_path}/pv"
wind_netcdf_path = f"{wind_path}/wind_data.nc"
wind_metadata_path = f"{wind_path}/wind_metadata.csv"
pv_netcdf_path = f"{pv_path}/pv_data.nc"
pv_metadata_path = f"{pv_path}/pv_metadata.csv"
satellite_path = f"{root_data_path}/satellite.zarr"

"""
Fixtures for testing
"""


import datetime as dt
import logging
import os

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.sqlmodels import Base, SiteSQL, GenerationSQL
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def engine():
    """Database engine fixture."""

    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url
        engine = create_engine(url)

        yield engine

        engine.dispose()


@pytest.fixture()
def db_conn(engine):
    """Create db connections and create/drop tables at the start/end of each test"""

    connection = DatabaseConnection(engine.url, echo=False)
    engine = connection.engine

    Base.metadata.create_all(engine)

    yield connection

    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_conn, engine):
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    with db_conn.get_session() as session:
        # begin the nested transaction
        session.begin()
        # use the connection with the already started transaction
        yield session

        # roll back the broader transaction
        session.rollback()


@pytest.fixture()
def sites(db_session):
    """Seed some initial data into DB."""

    sites = []
    # PV site
    site = SiteSQL(
        client_site_id=1,
        latitude=20.59,
        longitude=78.96,
        capacity_kw=4,
        ml_id=1,
        asset_type="pv",
        country="india",
    )
    db_session.add(site)
    sites.append(site)

    # Wind site
    site = SiteSQL(
        client_site_id=2,
        latitude=20.59,
        longitude=78.96,
        capacity_kw=4,
        ml_id=2,
        asset_type="wind",
        country="india",
    )
    db_session.add(site)
    sites.append(site)

    db_session.commit()

    return sites


@pytest.fixture()
def generation_db_values(db_session, sites, init_timestamp):
    """Create some fake generations"""
    start_times = [init_timestamp - dt.timedelta(minutes=x*3+6) for x in range(10)]

    all_generations = []

    for site in sites:
        for i in range(0, 10):
            generation = GenerationSQL(
                site_uuid=site.site_uuid,
                generation_power_kw=i,
                start_utc=start_times[i],
                end_utc=start_times[i] + dt.timedelta(minutes=3),
            )
            all_generations.append(generation)

    db_session.add_all(all_generations)
    db_session.commit()

    return all_generations


@pytest.fixture()
def forecast_values():
    """Dummy forecast values"""

    n = 10  # number of forecast values
    step = 15  # in minutes
    init_utc = dt.datetime.now(dt.timezone.utc)
    start_utc = [init_utc + dt.timedelta(minutes=i * step) for i in range(n)]
    end_utc = [d + dt.timedelta(minutes=step) for d in start_utc]
    forecast_power_kw = [i * 10 for i in range(n)]
    forecast_values = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "forecast_power_kw": forecast_power_kw
    }

    return forecast_values


@pytest.fixture(scope="session")
def init_timestamp():
    """Returns a datetime floored to the last 15 mins"""

    return pd.Timestamp(dt.datetime.now(tz=None)).floor(dt.timedelta(minutes=15))


@pytest.fixture(scope="session")
def time_before_present():
    """Returns a fixed time in the past with specified offset"""

    now = pd.Timestamp.now(tz=None)

    def _time_before_present(dt: dt.timedelta):
        return now - dt

    return _time_before_present


@pytest.fixture(scope="session")
def nwp_data(tmp_path_factory, time_before_present):
    """Dummy NWP data"""

    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp.zarr"
    )

    # Last t0 to at least 2 hours ago and floor to 3-hour interval
    t0_datetime_utc = (time_before_present(dt.timedelta(hours=0))
                       .floor(dt.timedelta(hours=3)))
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=3),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds["ecmwf"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # log.error(ds.attrs)

    # TODO ds.attrs["_data_attrs"] isn't defined, check this isn't a problem
    # Add stored attributes to DataArray
    # ds.ecmwf.attrs = ds.attrs["_data_attrs"]
    # del ds.attrs["_data_attrs"]

    # AS NWP data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    temp_nwp_path = f"{tmp_path_factory.mktemp('data')}/nwp.zarr"
    os.environ["NWP_ZARR_PATH"] = temp_nwp_path
    ds.to_zarr(temp_nwp_path)


@pytest.fixture(scope="session")
def wind_data(tmp_path_factory, time_before_present):
    """Dummy wind data"""

    # AS wind data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    root_path = tmp_path_factory.mktemp('data')

    root_source_path = os.path.dirname(os.path.abspath(__file__))

    netcdf_source_path = f"{root_source_path}/test_data/wind/wind_data.nc"
    temp_netcdf_path = f"{root_path}/wind_data.nc"
    os.environ["WIND_NETCDF_PATH"] = temp_netcdf_path
    ds = xr.open_dataset(netcdf_source_path)

    # Set t0 to at least 2 hours ago and floor to 15-min interval
    # t0_datetime_utc = (time_before_present(dt.timedelta(hours=0))
    #                    .floor(dt.timedelta(minutes=15)))
    t0_datetime_utc = pd.Timestamp.now(tz=None).floor(dt.timedelta(minutes=15))
    ds.time_utc.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(minutes=15 * (len(ds.time_utc) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(minutes=15),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    # ds["wind"] = xr.DataArray(
    #     np.zeros([len(ds[c]) for c in ds.xindexes]),
    #     coords=[ds[c] for c in ds.xindexes],
    # )

    ds.to_netcdf(temp_netcdf_path, engine="h5netcdf")

    metadata_source_path = f"{root_source_path}/test_data/wind/wind_metadata.csv"
    temp_metadata_path = f"{root_path}/wind_metadata.csv"
    os.environ["WIND_METADATA_PATH"] = temp_metadata_path
    fs = fsspec.open(metadata_source_path).fs
    fs.copy(metadata_source_path, temp_metadata_path)

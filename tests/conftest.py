"""
Fixtures for testing
"""


import datetime as dt
import logging
import os
import random

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pvsite_datamodel import DatabaseConnection
from pvsite_datamodel.sqlmodels import Base, GenerationSQL, SiteSQL
from sqlalchemy import create_engine
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
        capacity_kw=20000,
        ml_id=1,
        asset_type="pv",
        country="india",
    )
    db_session.add(site)
    sites.append(site)

    # Wind site
    site = SiteSQL(
        client_site_id=2,
        latitude=26.4499,
        longitude=72.6399,
        capacity_kw=10000,
        ml_id=0,
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

    n = 100  # 5 hours of readings
    start_times = [init_timestamp - dt.timedelta(minutes=x*3) for x in range(n)]

    # remove some of the most recent readings (to simulate missing timestamps)
    del start_times[20]
    del start_times[8]
    del start_times[3]

    # Random power values in the range 0-10000kw
    power_values = [random.random()*10000 for _ in range(len(start_times))]

    all_generations = []
    for site in sites:
        for i in range(0, len(start_times)):
            generation = GenerationSQL(
                site_uuid=site.site_uuid,
                generation_power_kw=power_values[i],
                start_utc=start_times[i],
                end_utc=start_times[i] + dt.timedelta(minutes=3),
            )
            all_generations.append(generation)

    db_session.add_all(all_generations)
    db_session.commit()

    return all_generations

@pytest.fixture()
def generation_db_values_only_wind(db_session, sites, init_timestamp):
    """Create some fake generations"""

    n = 100  # 5 hours of readings
    start_times = [init_timestamp - dt.timedelta(minutes=x*3) for x in range(n)]

    # remove some of the most recent readings (to simulate missing timestamps)
    del start_times[20]
    del start_times[8]
    del start_times[3]

    # Random power values in the range 0-10000kw
    power_values = [random.random()*10000 for _ in range(len(start_times))]

    all_generations = []
    for site in sites:
        for i in range(0, len(start_times)):
            if site.asset_type.name == "wind":
                generation = GenerationSQL(
                    site_uuid=site.site_uuid,
                    generation_power_kw=power_values[i],
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
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp-no-data.zarr"
    )

    # Last t0 to at least 6 hours ago and floor to 12-hour interval
    t0_datetime_utc = (time_before_present(dt.timedelta(hours=0))
                    .floor('12h'))
    t0_datetime_utc = t0_datetime_utc - dt.timedelta(hours=6)
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=12 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=3),
    )

    # force lat and lon to be in 0.1 steps
    ds.latitude.values[:] = [35.0 - i*0.1 for i in range(len(ds.latitude))]
    ds.longitude.values[:] = [65.0 + i * 0.1 for i in range(len(ds.longitude))]

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

    # AS NWP data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    temp_nwp_path_ecmwf = f"{tmp_path_factory.mktemp('data')}/nwp_ecmwf.zarr"
    os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path_ecmwf
    ds.to_zarr(temp_nwp_path_ecmwf)

@pytest.fixture(scope="session")
def nwp_gfs_data(tmp_path_factory, time_before_present):
    """Dummy NWP data"""

    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp-no-data_gfs.zarr"
    )


    # Last t0 to at least 6 hours ago and floor to 3-hour interval
    t0_datetime_utc = (time_before_present(dt.timedelta(hours=0))
                    .floor('3h'))
    t0_datetime_utc = t0_datetime_utc - dt.timedelta(hours=6)
    ds.init_time_utc.values[:] = pd.date_range(
        t0_datetime_utc - dt.timedelta(hours=12 * (len(ds.init_time_utc) - 1)),
        t0_datetime_utc,
        freq=dt.timedelta(hours=3),
    )
    # force lat and lon to be in 0.1 steps
    ds.latitude.values[:] = [35.0 - i*0.1 for i in range(len(ds.latitude))]
    ds.longitude.values[:] = [65.0 + i * 0.1 for i in range(len(ds.longitude))]

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()


    ds["gfs"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # data_var = ds['gfs']

    # # Use .to_dataset() to split the data variable based on 'channel'
    # new_ds = data_var.to_dataset(dim='variable')    

    # AS NWP data is loaded by the app from environment variable,
    # save out data and set paths as environmental variables
    temp_nwp_path_gfs = f"{tmp_path_factory.mktemp('data')}/nwp_gfs.zarr"

    os.environ["NWP_GFS_ZARR_PATH"] = temp_nwp_path_gfs
    ds.to_zarr(temp_nwp_path_gfs)

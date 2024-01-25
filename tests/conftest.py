"""
Fixtures for testing
"""


import datetime as dt
import os

import pytest
from pvsite_datamodel.sqlmodels import Base, SiteSQL
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def engine():
    """Database engine fixture."""

    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url
        engine = create_engine(url)
        Base.metadata.create_all(engine)

        yield engine


@pytest.fixture()
def db_session(engine):
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    connection = engine.connect()
    # begin the nested transaction
    transaction = connection.begin()
    # use the connection with the already started transaction

    with Session(bind=connection) as session:
        yield session

        session.close()
        # roll back the broader transaction
        transaction.rollback()
        # put back the connection to the connection pool
        connection.close()
        session.flush()

    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def db_data(engine):
    """Seed some initial data into DB."""

    with engine.connect() as connection:
        with Session(bind=connection) as session:
            n_sites = 3

            # Sites
            for i in range(n_sites):
                site = SiteSQL(
                    client_site_id=i + 1,
                    latitude=51,
                    longitude=3,
                    capacity_kw=4,
                    ml_id=i,
                    country="india"
                )
                session.add(site)

            session.commit()


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

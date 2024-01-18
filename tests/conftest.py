"""
Fixtures for testing
"""


import pytest
from pvsite_datamodel.connection import DatabaseConnection
from pvsite_datamodel.sqlmodels import Base, SiteSQL
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session", autouse=True)
def db_conn():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:14.5") as postgres:
        url = postgres.get_connection_url()

        database_connection = DatabaseConnection(url, echo=False)
        engine = database_connection.engine

        Base.metadata.create_all(engine)

        yield database_connection

        engine.dispose()


@pytest.fixture()
def db_session(db_conn):
    """Creates a new database session for a test.

    We automatically roll back whatever happens when the test completes.
    """

    with db_conn.get_session() as session:
        with session.begin():
            yield session
            session.rollback()


@pytest.fixture(scope="session", autouse=True)
def db_data(db_conn):
    """Seed some initial data into DB."""

    with db_conn.get_session() as session:
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

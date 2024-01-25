"""
Script for seeding a local DB.
"""

import os

from pvsite_datamodel.connection import DatabaseConnection
from pvsite_datamodel.sqlmodels import Base
from pvsite_datamodel.write.user_and_site import (
    create_site,
)


def _confirm_action() -> bool:
    """
    Provides opportunity for user to decide whether to proceed with code execution
    """
    while True:
        confirm = input(
            "!! This script will drop all tables and \
recreate them before seeding. Are you sure you wish to proceed? [y]Yes or [n]No: "
        )
        if confirm.strip().lower() in ("y", "n"):
            return confirm == "y"
        print("\nInvalid Option. Please Enter a Valid Option.")


def seed_db():
    """
    Drops existing tables and recreated schema before seeding some dummy data
    """
    url = os.getenv("DB_URL")
    db_conn = DatabaseConnection(url=url, echo=False)

    # 1. Create tables
    engine = db_conn.engine
    if not _confirm_action():
        print("Quitting.")
        return

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with db_conn.get_session() as session:

        print("Seeding database")
        site, _ = create_site(
            session,
            client_site_id=1,
            client_site_name="dummy_site_1",
            latitude=0.0,
            longitude=0.0,
            capacity_kw=10.0,
            asset_type="pv",
            country="india",
        )

        site, _ = create_site(
            session,
            client_site_id=2,
            client_site_name="dummy_site_2",
            latitude=0.0,
            longitude=0.0,
            capacity_kw=10.0,
            asset_type="wind",
            country="india",
        )

        print("Database successfully seeded")

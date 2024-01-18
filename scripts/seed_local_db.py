"""
Script for seeding a local DB.
"""

import os

from pvsite_datamodel.connection import DatabaseConnection
from pvsite_datamodel.sqlmodels import Base
from pvsite_datamodel.write.user_and_site import (
    add_site_to_site_group,
    create_site,
    create_site_group,
    create_user,
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
        dummy_user = "dummy@openclimatefix.org"

        print("Seeding database")
        site_group = create_site_group(session, site_group_name="dummy_site_group")

        _ = create_user(
            session, email=dummy_user, site_group_name=site_group.site_group_name
        )

        site, _ = create_site(
            session,
            client_site_id=1234,
            client_site_name="dummy_site",
            latitude=0.0,
            longitude=0.0,
            capacity_kw=10.0,
            country="india",
        )

        add_site_to_site_group(
            session,
            site_uuid=site.site_uuid,
            site_group_name=site_group.site_group_name,
        )

        print("Database successfully seeded")

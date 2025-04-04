import datetime as dt
import pandas as pd
import os

from pvsite_datamodel.sqlmodels import SiteAssetType

from india_forecast_app.data.generation import get_generation_data, filter_on_sun_elevation


def test_filter_on_sun_elevation(sites):

    """Test for filtering generation data based on sun elevation"""

    site = sites[0]
    generation_df = pd.DataFrame(
        data=[
            ["2023-10-01", 0.0],
            ["2023-10-01 10:00", 0.0],  # this one will get removed
            ["2023-10-01 11:00", 1.0],
            ["2023-10-01 20:00", 0.0],
        ],
        columns=["time_utc", "1"],
    )
    generation_df.set_index("time_utc", inplace=True)

    filter_generation_df = filter_on_sun_elevation(generation_df=generation_df, site=site)
    assert len(filter_generation_df) == 3
    assert filter_generation_df.index[0] == "2023-10-01"
    assert filter_generation_df.index[1] == "2023-10-01 11:00"
    assert filter_generation_df.index[2] == "2023-10-01 20:00"


def test_get_generation_data_wind(db_session, sites, generation_db_values, init_timestamp, client_ruvnl):
    """Test for correct generation data"""

    # Test only checks for wind data as solar data not ready yet
    gen_sites = [s for s in sites if s.asset_type == SiteAssetType.wind][0:1]  # 1 site
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    gen_df, gen_meta = gen_data["data"], gen_data["metadata"]

    # Check for 5 (non-null) generation values
    assert len(gen_df) == 5
    assert not gen_df["0"].isnull().any()  # 0 is the ml_id/system_id of the wind site

    # Check first and last timestamps are correct
    assert gen_df.index[0] == init_timestamp - dt.timedelta(hours=1)
    assert gen_df.index[-1] == init_timestamp

    # Check for expected metadata
    assert len(gen_meta) == 1


def test_get_generation_data_pv(db_session, sites, generation_db_values, init_timestamp, client_ruvnl):
    """Test for correct generation data"""

    # Test only checks for wind data as solar data not ready yet
    gen_sites = [s for s in sites if s.asset_type == SiteAssetType.pv][0:1]  # 1 site
    gen_data = get_generation_data(db_session, gen_sites, timestamp=init_timestamp)
    gen_df, gen_meta = gen_data["data"], gen_data["metadata"]

    # Check for 5 (non-null) generation values
    assert len(gen_df) == 5
    assert not gen_df["1"].isnull().any()  # 1 is the ml_id/system_id of the pv site

    # Check first and last timestamps are correct
    assert gen_df.index[0] == init_timestamp - dt.timedelta(hours=1)
    assert gen_df.index[-1] == init_timestamp

    # Check for expected metadata
    assert len(gen_meta) == 1

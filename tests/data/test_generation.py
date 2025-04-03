import datetime as dt

from pvsite_datamodel.sqlmodels import SiteAssetType

from india_forecast_app.data.generation import get_generation_data


def test_get_generation_data(db_session, sites, generation_db_values, init_timestamp):
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

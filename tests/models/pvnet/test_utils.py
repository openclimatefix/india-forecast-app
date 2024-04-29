from india_forecast_app.models.pvnet.utils import set_night_time_zeros
import numpy as np

from ocf_datapipes.batch import BatchKey


def test_set_night_time_zeros():
    # set up preds (1,14,7)
    preds = np.random.rand(1, 5, 7)

    # check that all values are positive
    assert np.all(preds > 0)

    # set up batch, last 3 sun elevations are negative, so should set these to zero
    batch = {
        BatchKey.pv_solar_elevation: np.array([[0, 1, 2, 3, 4, 5, 6, -7, -8, -9]]),
        BatchKey.pv_t0_idx: 4,
    }

    # test function
    preds = set_night_time_zeros(batch, preds)

    # check that all values are zero
    assert np.all(preds[:, 2:, :] == 0)
    # check that all values are positive
    assert np.all(preds[:, :2, :] > 0)

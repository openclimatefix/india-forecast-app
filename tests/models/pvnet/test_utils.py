""" Tests for utils for pvnet"""
import numpy as np
import os
from ocf_datapipes.batch import BatchKey
import tempfile

from india_forecast_app.models.pvnet.utils import set_night_time_zeros, save_batch


def test_set_night_time_zeros():
    """Test for setting night time zeros"""
    # set up preds (1,5,7) {example, time, plevels}
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


def test_save_batch():
    """ test to check batches are saved """

    # set up batch
    batch = {"key": "value"}
    i = 1
    model_name = "test_model_name"

    # create temp folder
    with tempfile.TemporaryDirectory() as temp_dir:
        save_batch(batch, i, model_name, save_batches_dir=temp_dir)

        # check that batch is saved
        assert os.path.exists(f"{temp_dir}/batch_{i}_{model_name}.pt")

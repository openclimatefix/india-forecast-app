"""
Model classes (currently just allows for loading a dummy model)
"""

import datetime as dt
import math
import random


class DummyModel:
    """
    Dummy model that emulates the capabilities expected by a real model
    """
    
    @property
    def version(self):
        """Version number"""
        return "0.0.0"

    def __init__(self):
        """Initializer for the model"""
        pass

    def predict(self, site_id: str, timestamp: dt.datetime):
        """Make a prediction for the model"""
        return self._generate_dummy_forecast(timestamp)

    def _generate_dummy_forecast(self, timestamp: dt.datetime):
        """Generates a fake 2-day forecast (15 minute intervals"""
        start = timestamp
        end = timestamp + dt.timedelta(days=2)
        step = dt.timedelta(minutes=15)
        numSteps = int((end - start) / step)
        values: list[dict] = []

        for i in range(numSteps):
            time = start + i * step
            _yield = _basicSolarYieldFunc(int(time.timestamp()))
            values.append(
                {
                    "start_utc": time,
                    "end_utc": time + step,
                    "forecast_power_kw": int(_yield),
                }
            )

        return values


def _basicSolarYieldFunc(timeUnix: int, scaleFactor: int = 10000) -> float:
    """Gets a fake solar yield for the input time.

    The basic yield function is built from a sine wave
    with a period of 24 hours, peaking at 12 hours.
    Further convolutions modify the value according to time of year.

    Args:
        timeUnix: The time in unix time.
        scaleFactor: The scale factor for the sine wave.
            A scale factor of 10000 will result in a peak yield of 10 kW.
    """
    # Create a datetime object from the unix time
    time = dt.datetime.fromtimestamp(timeUnix, tz=dt.UTC)
    # The functions x values are hours, so convert the time to hours
    hour = time.day * 24 + time.hour + time.minute / 60 + time.second / 3600

    # scaleX makes the period of the function 24 hours
    scaleX = math.pi / 12
    # translateX moves the minimum of the function to 0 hours
    translateX = -math.pi / 2
    # translateY modulates the base function based on the month.
    # * + 0.5 at the summer solstice
    # * - 0.5 at the winter solstice
    translateY = math.sin((math.pi / 6) * time.month + translateX) / 2.0

    # basefunc ranges between -1 and 1 with a period of 24 hours,
    # peaking at 12 hours.
    # translateY changes the min and max to range between 1.5 and -1.5
    # depending on the month.
    basefunc = math.sin(scaleX * hour + translateX) + translateY
    # Remove negative values
    basefunc = max(0, basefunc)
    # Steepen the curve. The divisor is based on the max value
    basefunc = basefunc**4 / 1.5**4

    # Instead of completely random noise, apply based on the following process:
    # * A base noise function which is the product of long and short sines
    # * The resultant function modulates with very small amplitude around 1
    noise = (math.sin(math.pi * time.hour) / 20) * (
        math.sin(math.pi * time.hour / 3)
    ) + 1
    noise = noise * random.random() / 20 + 0.97

    # Create the output value from the base function, noise, and scale factor
    output = basefunc * noise * scaleFactor

    return output


def _basicWindYieldFunc(timeUnix: int, scaleFactor: int = 10000) -> float:
    """Gets a fake wind yield for the input time."""
    output = min(scaleFactor, scaleFactor * 10 * random.random())

    return output

import numpy as np
import pandas as pd

from src.preprocessing.telemetry import TelemetryExtractor


class _FakeGPSStream:
    metadata = {"fourcc": "GPS5"}

    def to_dataframe(self, include_quality=False):
        assert include_quality
        return pd.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0, 3.0],
                "lat": [0.0, 37.0, 37.0001, 37.0002],
                "lon": [0.0, -122.0, -122.0001, -122.0002],
                "alt": [0.0, 10.0, 11.0, 12.0],
                "gps_fix": [0, 2, 3, 3],
                "gps_error_m": [99.0, 2.0, 1.0, 40.0],
                "valid": [False, True, True, True],
            }
        )


class _FakeTelemetry:
    def get_stream(self, name):
        if name in {"GPS", "GPS5"}:
            return _FakeGPSStream()
        raise KeyError(name)


def test_extract_gps_retains_quality_and_rejects_bad_samples():
    extractor = object.__new__(TelemetryExtractor)
    extractor.min_gps_fix = 3
    extractor.max_gps_accuracy_m = 20.0
    extractor.max_gps_position_dop = 5.0

    track = extractor._extract_gps(_FakeTelemetry())

    assert track is not None
    np.testing.assert_allclose(track.latitudes, [37.0001])
    np.testing.assert_allclose(track.fixes, [3.0])
    np.testing.assert_allclose(track.accuracies, [1.0])
    assert track.position_dops is None


def test_telemetry_dataframes_have_stable_persistence_columns():
    from src.core.types import GPSTrack, IMUData

    gps = GPSTrack(
        latitudes=np.array([37.0]),
        longitudes=np.array([-122.0]),
        altitudes=np.array([10.0]),
        timestamps=np.array([1.0]),
        accuracies=np.array([2.0]),
        fixes=np.array([3.0]),
        position_dops=np.array([1.2]),
    )
    imu = IMUData(
        accelerometer=np.array([[1.0, 2.0, 3.0]]),
        gyroscope=np.array([[0.1, 0.2, 0.3]]),
        timestamps=np.array([1.0]),
    )

    assert {
        "timestamp_s",
        "latitude_deg",
        "longitude_deg",
        "fix_type",
        "horizontal_accuracy_m",
        "hdop",
        "ellipsoidal_height_m",
    } <= set(gps.to_dataframe().columns)
    assert {
        "timestamp_s",
        "accel_x_mps2",
        "gyro_z_radps",
    } <= set(imu.to_dataframe().columns)

from pathlib import Path
import os

import pytest

from src.preprocessing.telemetry import TelemetryExtractor


@pytest.mark.skipif(
    "TELEMETRY_TEST_VIDEO" not in os.environ,
    reason="Set TELEMETRY_TEST_VIDEO to a GoPro video path to run this test.",
)
def test_extract_telemetry_prints():
    video_path = Path(os.environ["TELEMETRY_TEST_VIDEO"])
    extractor = TelemetryExtractor()

    gps_track, imu_data = extractor.extract(video_path)

    print("GPS track:", gps_track)
    if gps_track is not None:
        print("GPS lat head:", gps_track.latitudes[:5])
        print("GPS lon head:", gps_track.longitudes[:5])
        if gps_track.altitudes is not None:
            print("GPS alt head:", gps_track.altitudes[:5])
        if gps_track.timestamps is not None:
            print("GPS ts head:", gps_track.timestamps[:5])

    print("IMU data:", imu_data)
    if imu_data is not None:
        print("IMU accel head:", imu_data.accelerometer[:5])
        print("IMU gyro head:", imu_data.gyroscope[:5])
        print("IMU ts head:", imu_data.timestamps[:5])

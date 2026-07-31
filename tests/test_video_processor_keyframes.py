import json

import cv2
import numpy as np
import pytest

from src.core.types import GPSTrack, IMUData, VideoInput
from src.preprocessing.video_processor import (
    KeyframeSelectionConfig,
    VideoProcessor,
)


def _processor(config: KeyframeSelectionConfig) -> VideoProcessor:
    processor = object.__new__(VideoProcessor)
    processor.keyframe_config = config
    processor.fps = config.candidate_fps
    processor.jpeg_quality = 10
    processor.configured_capture_metadata = {}
    return processor


def _checkerboard(size: int = 96) -> np.ndarray:
    y, x = np.indices((size, size))
    image = (20 + ((x // 8 + y // 8) % 2) * 215).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_keyframe_config_rejects_invalid_modes_and_rates():
    with pytest.raises(ValueError, match="keyframe mode"):
        KeyframeSelectionConfig.from_mapping(10, {"mode": "random"})
    with pytest.raises(ValueError, match="cannot exceed"):
        KeyframeSelectionConfig.from_mapping(
            10, {"candidate_fps": 2, "target_fps": 3}
        )


def test_fixed_rate_selection_preserves_source_indices_and_timestamps(tmp_path):
    config = KeyframeSelectionConfig.from_mapping(
        4,
        {
            "mode": "fixed_rate",
            "target_fps": 2,
            "min_sharpness": 0,
        },
    )
    processor = _processor(config)
    for index in range(4):
        cv2.imwrite(str(tmp_path / f"frame_{index + 1:04d}.jpg"), _checkerboard())

    source_timestamps = np.array([0.0, 0.24, 0.51, 0.76])
    records = processor._select_keyframes(
        tmp_path,
        source_timestamps=source_timestamps,
        source_fps=4.0,
        imu_data=None,
        gps_track=None,
    )

    selected = [record for record in records if record["selected"]]
    assert [record["candidate_index"] for record in selected] == [0, 2]
    assert [record["source_frame_index"] for record in selected] == [0, 2]
    assert [record["timestamp_s"] for record in selected] == [0.0, 0.51]
    assert selected[1]["selection_reasons"] == ["fixed_rate"]


def test_quality_motion_selection_records_visual_imu_gps_and_rejections(tmp_path):
    config = KeyframeSelectionConfig.from_mapping(
        2,
        {
            "mode": "quality_motion",
            "target_fps": 0.2,
            "min_interval_s": 0.25,
            "max_interval_s": 5.0,
            "min_sharpness": 20,
            "min_motion_fraction": 0.005,
            "min_angular_speed_rad_s": 0.5,
        },
    )
    processor = _processor(config)
    base = _checkerboard()
    shifted = np.roll(base, 6, axis=1)
    images = [base, base, shifted, np.full_like(base, 255)]
    for index, image in enumerate(images):
        cv2.imwrite(str(tmp_path / f"frame_{index + 1:04d}.jpg"), image)

    timestamps = np.array([0.0, 0.5, 1.0, 1.5])
    gyro = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    imu = IMUData(
        accelerometer=np.tile([0.0, 0.0, 9.81], (4, 1)),
        gyroscope=gyro,
        timestamps=timestamps,
    )
    gps = GPSTrack(
        latitudes=np.array([37.0, 37.1]),
        longitudes=np.array([-122.0, -122.1]),
        altitudes=np.array([10.0, 11.0]),
        timestamps=np.array([0.0, 1.0]),
        fixes=np.array([3.0, 3.0]),
    )

    records = processor._select_keyframes(
        tmp_path,
        source_timestamps=timestamps,
        source_fps=2.0,
        imu_data=imu,
        gps_track=gps,
    )

    assert records[0]["selection_reasons"] == ["initial"]
    assert records[1]["selected"]
    assert "angular_motion" in records[1]["selection_reasons"]
    assert records[2]["selected"]
    assert "visual_motion" in records[2]["selection_reasons"]
    assert not records[3]["selected"]
    assert "exposure_rejected" in records[3]["selection_reasons"]
    assert records[2]["telemetry"]["gps"]["latitude_deg"] == 37.1
    assert records[1]["telemetry"]["angular_speed_rad_s"] == 1.0


def test_manifests_and_video_input_retain_exact_selected_identity(tmp_path):
    processor = _processor(KeyframeSelectionConfig())
    records = [
        {
            "candidate_index": 2,
            "source_frame_index": 60,
            "timestamp_s": 2.002,
            "selected": True,
            "selection_reasons": ["visual_motion"],
        }
    ]
    processor._save_keyframe_manifest(tmp_path, records)
    processor._save_capture_metadata(
        tmp_path,
        {
            "schema_version": 1,
            "video": {"width_px": 3840, "height_px": 2160},
            "calibration": {
                "lens_mode": "linear",
                "hypersmooth": "on",
            },
        },
    )

    manifest = processor.load_keyframe_manifest(tmp_path)
    capture = processor.load_capture_metadata(tmp_path)
    assert manifest["candidates"][0]["source_frame_index"] == 60
    assert capture["calibration"]["hypersmooth"] == "on"
    assert json.loads((tmp_path / "keyframes.json").read_text())["schema_version"] == 1

    video_input = VideoInput(
        video_path=tmp_path / "capture.mp4",
        image_dir=tmp_path,
        fps=2.0,
        frame_count=1,
        source_frame_indices=np.array([60]),
        frame_timestamps=np.array([2.002]),
    )
    np.testing.assert_array_equal(video_input.get_source_frame_indices(), [60])
    np.testing.assert_allclose(video_input.get_frame_timestamps(), [2.002])

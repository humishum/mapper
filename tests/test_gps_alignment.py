import numpy as np

from src.alignment.gps_aligner import GPSAligner
from src.core.types import (
    AlignmentStatus,
    CameraPoses,
    GPSTrack,
    PointCloud,
    PoseConvention,
)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _enu_to_geodetic(enu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pyproj import Transformer

    latitude, longitude, altitude = 37.4, -122.1, 125.0
    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    to_geodetic = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    x0, y0, z0 = to_ecef.transform(longitude, latitude, altitude)
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    ecef_to_enu = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
    ecef = enu @ ecef_to_enu + np.array([x0, y0, z0])
    lon_out, lat_out, alt_out = to_geodetic.transform(
        ecef[:, 0], ecef[:, 1], ecef[:, 2]
    )
    return np.asarray(lat_out), np.asarray(lon_out), np.asarray(alt_out)


def _synthetic_inputs(
    *,
    scale: float = 2.4,
    clock_offset_s: float = 1.4,
    gps_outlier: bool = True,
) -> tuple[PointCloud, CameraPoses, GPSTrack, np.ndarray]:
    timestamps = np.linspace(0.0, 30.0, 151)
    source = np.column_stack(
        (
            0.18 * timestamps + 1.2 * np.sin(0.65 * timestamps),
            0.08 * timestamps + 0.7 * np.cos(0.31 * timestamps),
            0.12 * np.sin(0.27 * timestamps),
        )
    )
    rotation = _rotation_z(33.0)
    target = (scale * rotation @ source.T).T + np.array([15.0, -7.0, 2.0])
    if gps_outlier:
        target[75] += np.array([35.0, -20.0, 10.0])
    lat, lon, alt = _enu_to_geodetic(target)

    poses = np.repeat(np.eye(4)[None, :, :], len(source), axis=0)
    poses[:, :3, 3] = source
    camera_poses = CameraPoses(poses=poses, timestamps=timestamps)
    gps_track = GPSTrack(
        latitudes=lat,
        longitudes=lon,
        altitudes=alt,
        timestamps=timestamps + clock_offset_s,
        accuracies=np.ones(len(source)),
        fixes=np.full(len(source), 3.0),
        position_dops=np.full(len(source), 1.2),
    )
    pointcloud = PointCloud(points=source.astype(np.float32))
    return pointcloud, camera_poses, gps_track, rotation


def test_timestamped_robust_sim3_recovers_clock_scale_and_rotation():
    pointcloud, poses, gps, expected_rotation = _synthetic_inputs()
    aligner = GPSAligner(max_clock_offset_s=3.0, clock_step_s=0.1, max_rmse_m=2.0)

    result = aligner.align(pointcloud, poses, gps, allow_scale=True)

    assert result.status == AlignmentStatus.ALIGNED
    assert result.aligned_pointcloud is not None
    np.testing.assert_allclose(result.clock_offset_s, 1.4, atol=0.11)
    np.testing.assert_allclose(result.scale, 2.4, rtol=0.02)
    recovered_rotation = result.transform[:3, :3] / result.scale
    np.testing.assert_allclose(recovered_rotation, expected_rotation, atol=0.02)
    assert result.rmse_m is not None and result.rmse_m < 0.15
    assert result.inlier_count < result.correspondence_count
    assert result.anchor_wgs84 is not None
    longitude, latitude, height = result.anchor_lon_lat_height
    assert -123.0 < longitude < -121.0
    assert 36.0 < latitude < 38.0
    assert result.to_dict()["anchor_lon_lat_height"] == [
        longitude,
        latitude,
        height,
    ]
    assert result.horizontal_rmse_m is not None
    assert result.vertical_rmse_m is not None
    assert np.linalg.norm(result.enu_to_ecef_transform[:3, 3]) > 6_000_000
    aligned_poses = result.transform_poses(poses)
    np.testing.assert_allclose(
        aligned_poses.get_positions(),
        (result.transform[:3, :3] @ poses.get_positions().T).T
        + result.transform[:3, 3],
        atol=1e-9,
    )
    assert aligned_poses.coordinate_frame.name == "artifact_local"
    assert aligned_poses.pose_convention == PoseConvention.CAMERA_TO_WORLD


def test_metric_model_locks_scale():
    pointcloud, poses, gps, _ = _synthetic_inputs(scale=1.0, gps_outlier=False)
    result = GPSAligner(max_clock_offset_s=3.0).align(
        pointcloud, poses, gps, allow_scale=False
    )

    assert result.accepted
    assert result.scale == 1.0


def test_gravity_constraint_is_part_of_similarity_solution_when_frame_is_known():
    pointcloud, poses, gps, expected_rotation = _synthetic_inputs(
        scale=1.0, gps_outlier=False
    )
    result = GPSAligner(max_clock_offset_s=3.0).align(
        pointcloud,
        poses,
        gps,
        allow_scale=False,
        gravity_direction_model=np.array([0.0, 0.0, -1.0]),
    )

    assert result.accepted
    assert result.method == "gps_gravity_constrained_weighted_umeyama"
    np.testing.assert_allclose(result.transform[:3, :3], expected_rotation, atol=0.02)
    assert result.diagnostics["gravity_constrained"] is True


def test_alignment_rejects_missing_timestamps_explicitly():
    pointcloud, poses, gps, _ = _synthetic_inputs(gps_outlier=False)
    poses.timestamps = None

    result = GPSAligner().align(pointcloud, poses, gps)

    assert result.status == AlignmentStatus.UNALIGNED
    assert result.reason == "pose_timestamps_missing"
    assert result.aligned_pointcloud is None
    np.testing.assert_array_equal(result.transform, np.eye(4))


def test_gps_quality_filter_and_robust_anchor_ignore_bad_fix():
    track = GPSTrack(
        latitudes=np.array([0.0, 37.0, 37.0001, 52.0]),
        longitudes=np.array([0.0, -122.0, -122.0001, 13.0]),
        altitudes=np.array([0.0, 10.0, 11.0, 100.0]),
        timestamps=np.arange(4.0),
        accuracies=np.array([50.0, 1.0, 1.0, 1.0]),
        fixes=np.array([0.0, 3.0, 3.0, 2.0]),
        position_dops=np.array([99.0, 1.0, 1.0, 1.0]),
    )

    filtered = track.filter_quality()
    anchor, ecef = filtered.robust_anchor()

    assert len(filtered) == 2
    assert 37.0 <= anchor[0] <= 37.0001
    assert -122.0001 <= anchor[1] <= -122.0
    assert ecef.shape == (3,)


def test_world_to_camera_pose_positions_are_explicit():
    c2w = np.eye(4)
    c2w[:3, 3] = [2.0, -3.0, 4.0]
    w2c = np.linalg.inv(c2w)[None, :, :]

    poses = CameraPoses(
        poses=w2c,
        pose_convention=PoseConvention.WORLD_TO_CAMERA,
    )

    np.testing.assert_allclose(poses.get_positions(), [[2.0, -3.0, 4.0]])
    assert {"fx", "fy", "cx", "cy"} <= set(poses.to_dataframe().columns)

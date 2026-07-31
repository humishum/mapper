from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.types import PointCloud, PoseConvention, VideoInput
from src.models import get_model, list_models
from src.models.da3_streaming import DA3StreamingModel
from src.models.mast3r_slam import MASt3RSLAMModel
from src.models.must3r import MUSt3RModel
from src.models.orb_slam import ORBSLAMModel
from src.models.vggt import VGGTModel
from src.models.vggt_long import VGGTLongModel
from src.models.vggt_omega import VGGTOmegaModel


def _video_input(tmp_path: Path, frame_count: int = 3) -> VideoInput:
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    return VideoInput(
        video_path=tmp_path / "capture.mp4",
        image_dir=image_dir,
        fps=2.0,
        frame_count=frame_count,
        source_frame_indices=np.array([10, 25, 41][:frame_count]),
        frame_timestamps=np.array([0.5, 1.25, 2.05][:frame_count]),
    )


def test_requested_adapters_are_registered_with_truthful_capabilities():
    assert {
        "da3_streaming",
        "vggt_long",
        "mast3r_slam",
        "vggt_omega",
    }.issubset(list_models())
    assert get_model("vggt_long") is VGGTLongModel
    assert get_model("mast3r_slam") is MASt3RSLAMModel
    assert get_model("vggt_omega") is VGGTOmegaModel

    for adapter in (
        DA3StreamingModel(),
        VGGTLongModel(),
        MASt3RSLAMModel(),
        VGGTOmegaModel(),
        VGGTModel(),
        MUSt3RModel(),
        ORBSLAMModel(),
    ):
        assert not adapter.outputs_metric_scale
    assert not DA3StreamingModel.outputs_confidence
    assert not VGGTLongModel.outputs_confidence
    assert not MASt3RSLAMModel.outputs_confidence
    assert VGGTOmegaModel.outputs_confidence


def test_external_adapter_revisions_are_explicit_full_commits():
    for adapter in (
        DA3StreamingModel(),
        VGGTLongModel(),
        MASt3RSLAMModel(),
        VGGTOmegaModel(),
    ):
        assert len(adapter.upstream_revision) == 40
        int(adapter.upstream_revision, 16)
        assert adapter.verify_upstream_revision
    vggt = VGGTModel()
    assert len(vggt.model_revision) == 40
    int(vggt.model_revision, 16)


def test_da3_imports_c2w_poses_with_exact_capture_identity(tmp_path):
    path = tmp_path / "camera_poses.txt"
    matrices = np.repeat(np.eye(4)[None], 3, axis=0)
    matrices[:, 0, 3] = [0.0, 1.0, 2.0]
    path.write_text(
        "\n".join(" ".join(map(str, matrix.reshape(-1))) for matrix in matrices)
    )
    adapter = DA3StreamingModel({"verify_upstream_revision": False})
    poses = adapter._load_poses(
        path,
        timestamps=np.array([0.5, 1.25, 2.05]),
        frame_indices=np.array([10, 25, 41]),
    )

    assert poses is not None
    assert poses.pose_convention == PoseConvention.CAMERA_TO_WORLD
    np.testing.assert_array_equal(poses.frame_indices, [10, 25, 41])
    np.testing.assert_allclose(poses.timestamps, [0.5, 1.25, 2.05])


def test_must3r_preserves_exact_capture_identity_for_all_scene_poses(tmp_path):
    video_input = _video_input(tmp_path)
    scene = SimpleNamespace(
        cams2world=np.repeat(np.eye(4)[None], 3, axis=0),
        focals=None,
    )

    poses = MUSt3RModel()._extract_poses(
        scene, video_input, frame_indices=[0, 1, 2]
    )

    assert poses is not None
    np.testing.assert_array_equal(poses.frame_indices, [10, 25, 41])
    np.testing.assert_allclose(poses.timestamps, [0.5, 1.25, 2.05])


def test_must3r_preserves_per_frame_focal_and_processed_shape(tmp_path):
    video_input = _video_input(tmp_path)
    scene = SimpleNamespace(
        cams2world=np.repeat(np.eye(4)[None], 3, axis=0),
        focals=[100.0, 110.0, 120.0],
        true_shape=np.array([[400, 512], [384, 512], [336, 512]]),
    )

    poses = MUSt3RModel()._extract_poses(
        scene, video_input, frame_indices=[0, 1, 2]
    )

    np.testing.assert_allclose(poses.intrinsics[:, 0, 0], [100, 110, 120])
    np.testing.assert_allclose(poses.intrinsics[:, 0, 2], [256, 256, 256])
    np.testing.assert_allclose(poses.intrinsics[:, 1, 2], [200, 192, 168])


def test_da3_combined_ply_is_relative_scale_and_required(tmp_path):
    adapter = DA3StreamingModel({"verify_upstream_revision": False})
    ply_path = tmp_path / "cloud.ply"
    PointCloud(points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32)).save_ply(
        ply_path
    )

    cloud = adapter._load_combined_ply(ply_path)
    assert not cloud.is_metric
    with pytest.raises(FileNotFoundError):
        adapter._load_combined_ply(tmp_path / "missing.ply")


def test_da3_load_validates_checkout_without_weights(tmp_path):
    checkout = tmp_path / "Depth-Anything-3"
    streaming = checkout / "da3_streaming"
    config = streaming / "configs" / "base_config.yaml"
    config.parent.mkdir(parents=True)
    (streaming / "da3_streaming.py").write_text("# entry point\n")
    config.write_text("Model: {}\n")

    adapter = DA3StreamingModel(
        {
            "da3_path": str(checkout),
            "verify_upstream_revision": False,
        }
    )
    adapter.load()
    assert adapter._is_loaded
    assert adapter.da3_dir == streaming.resolve()


def test_da3_runtime_config_resolves_mapper_owned_weights(tmp_path):
    weights_dir = tmp_path / "weights" / "da3_streaming"
    weights_dir.mkdir(parents=True)
    for filename in ("model.safetensors", "config.json", "dino_salad.ckpt"):
        (weights_dir / filename).write_bytes(b"checkpoint")
    config = tmp_path / "base.yaml"
    config.write_text(
        "Weights:\n"
        f"  DA3: {weights_dir / 'model.safetensors'}\n"
        f"  DA3_CONFIG: {weights_dir / 'config.json'}\n"
        f"  SALAD: {weights_dir / 'dino_salad.ckpt'}\n"
        "Model:\n"
        "  align_method: sim3\n"
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    adapter = DA3StreamingModel(
        {
            "da3_config_path": str(config),
            "verify_upstream_revision": False,
        }
    )

    runtime = adapter._materialize_runtime_config(output_dir)

    resolved = runtime.read_text()
    assert str(weights_dir / "model.safetensors") in resolved
    assert "align_method: sim3" in resolved


def test_default_in_process_model_weights_live_under_repo_weights():
    for adapter, attribute in (
        (MUSt3RModel(), "weights_path"),
        (VGGTModel(), "weights_path"),
        (VGGTOmegaModel(), "checkpoint_path"),
    ):
        path = Path(getattr(adapter, attribute))
        assert not path.is_absolute()
        assert path.parts[0] == "weights"


def test_vggt_long_parses_outputs_with_exact_capture_identity(tmp_path):
    pose_path = tmp_path / "camera_poses.txt"
    pose_path.write_text(" ".join(map(str, np.eye(4).reshape(-1))))
    poses = VGGTLongModel._load_poses(
        pose_path,
        np.array([3.25]),
        np.eye(3, dtype=np.float32)[None],
        np.array([87]),
    )

    assert poses.pose_convention == PoseConvention.CAMERA_TO_WORLD
    np.testing.assert_array_equal(poses.frame_indices, [87])
    np.testing.assert_allclose(poses.timestamps, [3.25])


def test_mast3r_slam_maps_keyframe_ids_to_exact_source_frames(tmp_path):
    trajectory = tmp_path / "frames.txt"
    trajectory.write_text(
        "0 1 2 3 0 0 0 1\n"
        "0.0666666667 4 5 6 0 0 0 1\n"
    )
    video_input = _video_input(tmp_path)

    poses = MASt3RSLAMModel._load_trajectory(trajectory, video_input)

    assert poses.pose_convention == PoseConvention.CAMERA_TO_WORLD
    np.testing.assert_array_equal(poses.frame_indices, [10, 41])
    np.testing.assert_allclose(poses.timestamps, [0.5, 2.05])
    np.testing.assert_allclose(poses.poses[:, :3, 3], [[1, 2, 3], [4, 5, 6]])


def test_mast3r_slam_exposes_selected_jpegs_as_png_symlinks(tmp_path):
    sources = []
    for index in range(2):
        source = tmp_path / f"frame_{index:08d}.jpg"
        source.write_bytes(b"jpeg")
        sources.append(source)
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    MASt3RSLAMModel._prepare_png_input(sources, input_dir)

    links = sorted(input_dir.glob("*.png"))
    assert [link.name for link in links] == [
        "frame_00000000.png",
        "frame_00000001.png",
    ]
    assert all(link.is_symlink() for link in links)


def test_mast3r_slam_command_is_headless_and_uses_requested_output(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "MASt3R-SLAM"
    config = checkout / "config" / "base.yaml"
    config.parent.mkdir(parents=True)
    (checkout / "main.py").write_text("# entry point\n")
    config.write_text("dataset: {}\n")
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    output_dir = tmp_path / "out"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.models.mast3r_slam.subprocess.run", fake_run)
    adapter = MASt3RSLAMModel(
        {
            "mast3r_slam_path": str(checkout),
            "verify_upstream_revision": False,
        }
    )
    adapter.load()
    adapter._run_mast3r_slam(image_dir, output_dir)

    command, kwargs = calls[-1]
    assert "--no-viz" in command
    assert command[command.index("--save-as") + 1] == str(output_dir)
    assert kwargs["cwd"] == checkout.resolve()
    assert kwargs["check"] is True


def test_vggt_omega_builds_relative_confident_cloud_without_model_package():
    adapter = VGGTOmegaModel(
        {
            "confidence_percentile": 0.0,
            "max_points": None,
            "verify_upstream_revision": False,
        }
    )
    images = np.array([[[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]]])
    predictions = {
        "depth": np.ones((1, 1, 1, 2, 1), dtype=np.float32),
        "depth_conf": np.array([[[[2.0, 3.0]]]], dtype=np.float32),
    }
    extrinsics = np.zeros((1, 1, 3, 4), dtype=np.float32)
    extrinsics[0, 0, :3, :3] = np.eye(3)
    intrinsics = np.array(
        [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]],
        dtype=np.float32,
    )

    cloud = adapter._build_pointcloud(
        images, predictions, extrinsics, intrinsics
    )

    assert not cloud.is_metric
    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.points, [[0, 0, 1], [1, 0, 1]])
    np.testing.assert_allclose(cloud.confidence, [2, 3])


def test_orb_adapter_is_disabled_by_default(tmp_path):
    adapter = ORBSLAMModel()
    video_input = _video_input(tmp_path)

    with pytest.raises(RuntimeError, match="disabled"):
        adapter.reconstruct(video_input, tmp_path / "out")

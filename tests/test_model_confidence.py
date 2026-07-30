from types import SimpleNamespace

import numpy as np

from src.models.must3r import MASt3RModel


def test_must3r_extracts_confidence_without_ply_roundtrip():
    model = MASt3RModel({"min_confidence": 2.0})
    scene = SimpleNamespace(
        x_out=[
            {
                "pts3d": np.array(
                    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
                    dtype=np.float32,
                ),
                "conf": np.array([[1.5, 3.5]], dtype=np.float32),
            }
        ],
        imgs=[
            np.array(
                [[[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]]],
                dtype=np.float32,
            )
        ],
    )

    cloud = model._extract_pointcloud(scene)

    np.testing.assert_array_equal(
        cloud.points, np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        cloud.colors, np.array([[255, 0, 63]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(cloud.confidence, np.array([3.5], dtype=np.float32))


def test_must3r_accepts_channel_first_images():
    model = MASt3RModel({"min_confidence": 0.0})
    scene = SimpleNamespace(
        x_out=[
            {
                "pts3d": np.zeros((1, 2, 3), dtype=np.float32),
                "conf": np.ones((1, 2), dtype=np.float32),
            }
        ],
        imgs=[np.ones((3, 1, 2), dtype=np.float32)],
    )

    cloud = model._extract_pointcloud(scene)

    assert cloud.colors.shape == (2, 3)
    assert cloud.confidence.shape == (2,)

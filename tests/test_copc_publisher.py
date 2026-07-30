from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

from src.publisher import (  # noqa: E402
    CopcPublisher,
    CopcPublisherConfig,
    LasStagingConfig,
    VoxelConsolidationConfig,
    consolidate_laz_shards,
    migrate_legacy_ply_to_copc,
    write_laz_shard,
)
from src.publisher.copc import (  # noqa: E402
    COPC_CONVERTER_ENV,
    COPC_CONVERTER_VERSION,
    REPO_TOOL_PATH,
)
from src.publisher.copc_validation import inspect_copc  # noqa: E402


def _converter() -> Path | None:
    configured = os.environ.get(COPC_CONVERTER_ENV)
    if configured and Path(configured).is_file():
        return Path(configured)
    if REPO_TOOL_PATH.is_file():
        return REPO_TOOL_PATH
    return None


def _arrays(seed: int, count: int, x_offset: float = 0.0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3)).astype(np.float32)
    points[:, 0] += x_offset
    colors = rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
    confidence = rng.random(count, dtype=np.float32)
    contributors = rng.integers(1, 20, size=count, dtype=np.uint16)
    return points, colors, confidence, contributors


def test_laz_staging_writes_canonical_schema_in_chunks(tmp_path: Path):
    points, colors, confidence, contributors = _arrays(4, 5_003)
    source_indices = np.arange(len(points), dtype=np.uint16) % 3
    output = tmp_path / "source.laz"

    result = write_laz_shard(
        output,
        points,
        colors,
        source_indices,
        confidence=confidence,
        contributor_count=contributors,
        config=LasStagingConfig(chunk_points=997),
    )

    assert result.point_count == len(points)
    assert result.source_indices == (0, 1, 2)
    assert result.file_bytes == output.stat().st_size
    assert len(result.sha256) == 64
    with laspy.open(output) as reader:
        assert str(reader.header.version) == "1.4"
        assert reader.header.point_format.id == 7
        assert tuple(reader.header.scales) == (0.001, 0.001, 0.001)
        assert tuple(reader.header.offsets) == (0.0, 0.0, 0.0)
        assert "Confidence" in reader.header.point_format.dimension_names
        assert "ContributorCount" in reader.header.point_format.dimension_names
        staged = reader.read()

    np.testing.assert_array_equal(staged.point_source_id, source_indices)
    np.testing.assert_array_equal(staged["Confidence"], confidence)
    np.testing.assert_array_equal(staged["ContributorCount"], contributors)
    np.testing.assert_array_equal(staged.red, colors[:, 0].astype(np.uint16) * 257)
    np.testing.assert_array_equal(staged.green, colors[:, 1].astype(np.uint16) * 257)
    np.testing.assert_array_equal(staged.blue, colors[:, 2].astype(np.uint16) * 257)


def test_laz_staging_requires_finite_confidence(tmp_path: Path):
    points, colors, confidence, _ = _arrays(2, 10)
    confidence[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_laz_shard(
            tmp_path / "bad.laz",
            points,
            colors,
            1,
            confidence=confidence,
        )


def test_laz_staging_supports_geometry_without_rgb_or_confidence(tmp_path: Path):
    points, _, _, contributors = _arrays(9, 321)
    output = tmp_path / "geometry-only.laz"

    write_laz_shard(
        output,
        points,
        None,
        8,
        contributor_count=contributors,
    )

    with laspy.open(output) as reader:
        assert reader.header.point_format.id == 6
        dimensions = tuple(reader.header.point_format.dimension_names)
        assert "red" not in dimensions
        assert "Confidence" not in dimensions
        assert "ContributorCount" in dimensions
        staged = reader.read()
    np.testing.assert_array_equal(staged["ContributorCount"], contributors)


def test_voxel_consolidation_crosses_shards_and_uses_highest_confidence(
    tmp_path: Path,
):
    first = tmp_path / "first.laz"
    second = tmp_path / "second.laz"
    # The first point in each shard occupies the same 2 cm voxel. The second
    # shard's higher-confidence observation must own the consolidated record.
    write_laz_shard(
        first,
        np.array([[0.001, 0.002, 0.003], [1.0, 0.0, 0.0]]),
        np.array([[10, 20, 30], [1, 2, 3]], dtype=np.uint8),
        4,
        confidence=np.array([0.2, 0.7], dtype=np.float32),
        contributor_count=np.array([2, 1], dtype=np.uint16),
    )
    write_laz_shard(
        second,
        np.array([[0.019, 0.018, 0.017], [2.0, 0.0, 0.0]]),
        np.array([[200, 150, 100], [4, 5, 6]], dtype=np.uint8),
        9,
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        contributor_count=np.array([3, 1], dtype=np.uint16),
    )

    result = consolidate_laz_shards(
        [first, second],
        tmp_path / "consolidated",
        config=VoxelConsolidationConfig(
            voxel_size=0.02,
            max_bucket_points=10,
            read_chunk_points=2,
            temp_dir=tmp_path,
        ),
    )

    assert result.points_in == 4
    assert result.points_out == 3
    assert result.contributor_total == 7
    records = []
    for shard in result.shards:
        with laspy.open(shard) as reader:
            records.append(reader.read())
    sources = np.concatenate([np.asarray(item.point_source_id) for item in records])
    confidence = np.concatenate([np.asarray(item["Confidence"]) for item in records])
    contributors = np.concatenate(
        [np.asarray(item["ContributorCount"]) for item in records]
    )
    red = np.concatenate([np.asarray(item.red) for item in records])
    winner = np.flatnonzero(np.isclose(confidence, 0.9))
    assert len(winner) == 1
    assert sources[winner[0]] == 9
    assert contributors[winner[0]] == 5
    assert red[winner[0]] == 200 * 257


def test_voxel_consolidation_without_confidence_uses_lowest_source(
    tmp_path: Path,
):
    for source, coordinate in ((12, 0.001), (2, 0.019)):
        write_laz_shard(
            tmp_path / f"{source}.laz",
            np.array([[coordinate, 0.0, 0.0]]),
            None,
            source,
            contributor_count=1,
        )

    result = consolidate_laz_shards(
        [tmp_path / "12.laz", tmp_path / "2.laz"],
        tmp_path / "without-confidence",
        config=VoxelConsolidationConfig(
            voxel_size=0.02,
            max_bucket_points=4,
            read_chunk_points=1,
            temp_dir=tmp_path,
        ),
    )

    assert result.points_in == 2
    assert result.points_out == 1
    with laspy.open(result.shards[0]) as reader:
        dimensions = tuple(reader.header.point_format.dimension_names)
        consolidated = reader.read()
    assert "Confidence" not in dimensions
    assert consolidated.point_source_id[0] == 2
    assert consolidated["ContributorCount"][0] == 2


@pytest.mark.skipif(
    _converter() is None,
    reason=(
        "install the pinned converter with scripts/install_copc_converter.py "
        f"or set {COPC_CONVERTER_ENV}"
    ),
)
def test_publisher_merges_shards_and_preserves_attributes(tmp_path: Path):
    expected_confidence = []
    expected_contributors = []
    expected_colors = [[], [], []]
    shards = []
    expected_sources = {}
    for sequence, source_index in enumerate((17, 42)):
        points, colors, confidence, contributors = _arrays(
            80 + sequence, 12_000 + sequence * 137, x_offset=sequence * 15
        )
        shard = tmp_path / f"source-{source_index}.laz"
        write_laz_shard(
            shard,
            points,
            colors,
            source_index,
            confidence=confidence,
            contributor_count=contributors,
        )
        shards.append(shard)
        expected_sources[source_index] = len(points)
        expected_confidence.append(confidence)
        expected_contributors.append(contributors)
        for channel in range(3):
            expected_colors[channel].append(colors[:, channel].astype(np.uint16) * 257)

    output = tmp_path / "points.copc.laz"
    result = CopcPublisher(
        CopcPublisherConfig(
            executable=_converter(),
            memory_limit="256M",
            threads=2,
            temp_dir=tmp_path,
        )
    ).publish(shards, output)

    assert result.converter_version == COPC_CONVERTER_VERSION
    assert result.point_count == sum(expected_sources.values())
    assert result.source_distribution == expected_sources
    structure = inspect_copc(output)
    assert structure.point_count == result.point_count
    assert structure.hierarchy_point_count == result.point_count
    assert structure.root_present
    assert structure.all_nodes_reachable
    assert structure.hierarchy_pages == 1

    with laspy.open(output) as reader:
        published = reader.read()
    np.testing.assert_array_equal(
        np.sort(published["Confidence"]),
        np.sort(np.concatenate(expected_confidence)),
    )
    np.testing.assert_array_equal(
        np.sort(published["ContributorCount"]),
        np.sort(np.concatenate(expected_contributors)),
    )
    for actual, expected in zip(
        (published.red, published.green, published.blue),
        expected_colors,
    ):
        np.testing.assert_array_equal(
            np.sort(actual), np.sort(np.concatenate(expected))
        )


@pytest.mark.skipif(
    _converter() is None,
    reason=(
        "install the pinned converter with scripts/install_copc_converter.py "
        f"or set {COPC_CONVERTER_ENV}"
    ),
)
def test_publisher_supports_point_format_6_without_confidence(tmp_path: Path):
    points, _, _, contributors = _arrays(11, 4_000)
    shard = tmp_path / "geometry-only.laz"
    write_laz_shard(
        shard,
        points,
        None,
        6,
        contributor_count=contributors,
    )
    output = tmp_path / "geometry-only.copc.laz"

    result = CopcPublisher(
        CopcPublisherConfig(
            executable=_converter(),
            memory_limit="256M",
            threads=2,
            temp_dir=tmp_path,
        )
    ).publish(shard, output)

    assert result.structure.point_format == 6
    with laspy.open(output) as reader:
        dimensions = tuple(reader.header.point_format.dimension_names)
        assert "red" not in dimensions
        assert "Confidence" not in dimensions
        assert "ContributorCount" in dimensions


@pytest.mark.skipif(
    _converter() is None,
    reason=(
        "install the pinned converter with scripts/install_copc_converter.py "
        f"or set {COPC_CONVERTER_ENV}"
    ),
)
def test_legacy_ply_one_shot_migration(tmp_path: Path):
    from plyfile import PlyData, PlyElement

    count = 1_003
    vertices = np.zeros(
        count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = np.linspace(0, 3, count)
    vertices["red"] = np.arange(count) % 256
    input_ply = tmp_path / "legacy.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(input_ply)

    output = tmp_path / "legacy.copc.laz"
    result = migrate_legacy_ply_to_copc(
        input_ply,
        output,
        source_point_counts=[503, 500],
        max_points_per_shard=200,
        publisher_config=CopcPublisherConfig(
            executable=_converter(),
            memory_limit="256M",
            threads=2,
            temp_dir=tmp_path,
        ),
    )

    assert result.input_points == count
    assert result.output_points == count
    assert result.source_count == 2
    assert result.publish.source_distribution == {0: 503, 1: 500}
    assert output.is_file()
    assert not list(tmp_path.glob("mapper-ply-migration-*"))

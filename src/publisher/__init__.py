"""Canonical point-cloud publication.

The public API intentionally accepts LAS/LAZ shards rather than reconstruction
windows.  A shard can represent a window today, a SLAM submap later, or simply
an out-of-core batch.
"""

from .copc import (
    CopcPublisher,
    CopcPublisherConfig,
    CopcPublisherError,
    PublishResult,
)
from .las_staging import (
    LasShardResult,
    LasStagingConfig,
    write_laz_shard,
)
from .consolidation import (
    ConsolidationResult,
    VoxelConsolidationConfig,
    consolidate_laz_shards,
)
from .legacy_ply import (
    LegacyPlyStagingConfig,
    LegacyPlyStagingResult,
    source_counts_from_window_dir,
    stage_legacy_binary_ply,
)
from .migration import LegacyMigrationResult, migrate_legacy_ply_to_copc
from .onboarding import (
    ExistingCopcInspection,
    ExistingCopcOnboardingError,
    ExistingCopcPackager,
    ExistingCopcPackageResult,
    inspect_existing_copc,
    load_legacy_poses,
)
from .package import (
    PackageIdentity,
    PackagePublishResult,
    PackageSource,
    ReconstructionPackagePublisher,
    capture_id_for_file,
    package_source_from_window,
)

__all__ = [
    "CopcPublisher",
    "CopcPublisherConfig",
    "CopcPublisherError",
    "ConsolidationResult",
    "LasShardResult",
    "LasStagingConfig",
    "LegacyPlyStagingConfig",
    "LegacyPlyStagingResult",
    "LegacyMigrationResult",
    "ExistingCopcInspection",
    "ExistingCopcOnboardingError",
    "ExistingCopcPackager",
    "ExistingCopcPackageResult",
    "PackageIdentity",
    "PackagePublishResult",
    "PackageSource",
    "PublishResult",
    "ReconstructionPackagePublisher",
    "VoxelConsolidationConfig",
    "capture_id_for_file",
    "consolidate_laz_shards",
    "package_source_from_window",
    "stage_legacy_binary_ply",
    "source_counts_from_window_dir",
    "migrate_legacy_ply_to_copc",
    "inspect_existing_copc",
    "load_legacy_poses",
    "write_laz_shard",
]

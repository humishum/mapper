"""Deterministic fixtures for Phase 2 browser benchmarks."""

from .build_multisite import (
    CopcCapabilities,
    DEFAULT_SITES,
    MultiSiteFixture,
    SiteDefinition,
    build_multisite_fixture,
    inspect_copc_capabilities,
)

__all__ = [
    "CopcCapabilities",
    "DEFAULT_SITES",
    "MultiSiteFixture",
    "SiteDefinition",
    "build_multisite_fixture",
    "inspect_copc_capabilities",
]

"""Versioned feature-artifact contracts.

This package is intentionally independent of descriptor generators.  Its
schema-2 reader can be used from a new training process without RDKit, Torch,
or descriptor-provider imports.
"""

from .contracts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactContractError,
    artifact_content_identity,
    resolve_manifest_path,
    validate_artifact_manifest,
)
from .reader import (
    ArtifactReadError,
    FeatureArtifact,
    import_legacy_npz,
    load_feature_artifact,
    publish_feature_artifact,
    read_feature_manifest,
)
from .operations import (
    ArtifactInspection,
    ArtifactOperationError,
    DerivedOperation,
    DeriveResult,
    derive_features,
    inspect_feature_artifact,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactContractError",
    "artifact_content_identity",
    "resolve_manifest_path",
    "validate_artifact_manifest",
    "ArtifactReadError",
    "FeatureArtifact",
    "import_legacy_npz",
    "load_feature_artifact",
    "publish_feature_artifact",
    "read_feature_manifest",
    "ArtifactInspection",
    "ArtifactOperationError",
    "DerivedOperation",
    "DeriveResult",
    "derive_features",
    "inspect_feature_artifact",
]

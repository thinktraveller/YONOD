"""Public contracts for YONOD runtime and feature-operation configuration.

The package deliberately stops before descriptor generation and model
construction, so a bad YAML configuration can be rejected without importing
optional chemistry or machine-learning dependencies.
"""

from .contracts import (
    CONFIG_SCHEMA_VERSION,
    MODEL_PARAMETER_ROUTES,
    ConfigContractError,
    MISSING,
    resolve_config_path,
    validate_operation_config,
    validate_run_config,
)
from .loader import (
    ConfigLoadError,
    LoadedRunConfig,
    build_explicit_cli_overrides,
    load_operation_config,
    load_run_config,
    load_yaml_mapping,
    merge_config_layers,
    normalise_model_aliases,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "MODEL_PARAMETER_ROUTES",
    "ConfigContractError",
    "MISSING",
    "resolve_config_path",
    "validate_operation_config",
    "validate_run_config",
    "ConfigLoadError",
    "LoadedRunConfig",
    "build_explicit_cli_overrides",
    "load_operation_config",
    "load_run_config",
    "load_yaml_mapping",
    "merge_config_layers",
    "normalise_model_aliases",
]

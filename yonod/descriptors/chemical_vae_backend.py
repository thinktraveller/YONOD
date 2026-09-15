"""Verified encoder-only building blocks for the bundled Chemical VAE asset.

This module is deliberately not registered as a public YONOD descriptor.  It
contains the step-31.3 migration machinery only: an independent NumPy forward
reference, a PyTorch candidate, and strict HDF5/asset validation.  The public
``chemical_vae`` descriptor is introduced only after parity is demonstrated.

The supported source is the historical Keras 2 HDF5 *encoder* asset.  Decoder,
TerminalGRU, property heads, training data, and latent-space standardisation
are intentionally absent from this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


class ChemicalVaeAssetError(ValueError):
    """Raised when a historical asset is malformed or outside the supported contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - covered by optional-dependency callers.
        raise ChemicalVaeAssetError(
            "Chemical VAE encoder migration requires optional dependency h5py. "
            "Install the project requirements before selecting this descriptor."
        ) from exc
    return h5py


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - covered by optional-dependency callers.
        raise ChemicalVaeAssetError(
            "Chemical VAE encoder migration requires the existing optional PyTorch runtime."
        ) from exc
    return torch


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChemicalVaeAssetError(f"cannot read JSON asset {path}: {exc}") from exc


_LAYER_SEQUENCE: Tuple[Tuple[str, str], ...] = (
    ("input_molecule_smi", "InputLayer"),
    ("encoder_conv0", "Conv1D"),
    ("encoder_norm0", "BatchNormalization"),
    ("encoder_conv1", "Conv1D"),
    ("encoder_norm1", "BatchNormalization"),
    ("encoder_conv2", "Conv1D"),
    ("encoder_norm2", "BatchNormalization"),
    ("flatten_1", "Flatten"),
    ("encoder_dense0", "Dense"),
    ("dropout_1", "Dropout"),
    ("encoder_dense0_norm", "BatchNormalization"),
    ("z_mean_sample", "Dense"),
)

# A v5 conversion manifest is executable data.  Keep that loader narrow: it
# may load only the two source encoder byte identities which have their own
# direct-HDF5 TensorFlow and independent NumPy parity evidence.  In
# particular, the ``zinc_properties`` property-predictor head is not part of
# this map or this runtime; its separately trained *encoder* has its own hash.
_VERIFIED_ENCODER_SOURCES: Mapping[str, str] = {
    "zinc": "37e96cd3bc8f9680d3c2aa4d294a39e78f00a0decc22501bde23d5a464ac314a",
    "zinc_properties": "9f923d03c5ce558d1f0cb52d2a2f4e28f080e59ac5a8c81a4365d7b63ba9d1e2",
}
_CONVOLUTION_NAMES = ("encoder_conv0", "encoder_conv1", "encoder_conv2")
_BATCH_NORM_NAMES = ("encoder_norm0", "encoder_norm1", "encoder_norm2", "encoder_dense0_norm")
_DENSE_NAMES = ("encoder_dense0", "z_mean_sample")
_RUNTIME_LAYER_CONFIGS: Mapping[str, Mapping[str, Any]] = {
    "encoder_conv0": {"filters": 9, "kernel_size": (9,), "strides": (1,)},
    "encoder_conv1": {"filters": 9, "kernel_size": (9,), "strides": (1,)},
    "encoder_conv2": {"filters": 10, "kernel_size": (11,), "strides": (1,)},
    "encoder_norm0": {"epsilon": 0.001},
    "encoder_norm1": {"epsilon": 0.001},
    "encoder_norm2": {"epsilon": 0.001},
    "encoder_dense0_norm": {"epsilon": 0.001},
}


@dataclass(frozen=True)
class ChemicalVaeEncoderAsset:
    """Validated source metadata and raw Keras-layout weights for one encoder."""

    source_path: Path
    source_sha256: str
    variant: str
    max_len: int
    padding: str
    charset: Tuple[str, ...]
    hidden_dim: int
    keras_version: str
    keras_backend: str
    layer_configs: Mapping[str, Mapping[str, Any]]
    weights: Mapping[str, Mapping[str, np.ndarray]]

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.max_len, len(self.charset)

    @property
    def asset_id(self) -> str:
        return f"{self.variant}-{self.source_sha256[:16]}"

    def manifest_fragment(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "variant": self.variant,
            "source_encoder_sha256": self.source_sha256,
            "source_encoder_size_bytes": self.source_path.stat().st_size,
            "max_len": self.max_len,
            "padding": self.padding,
            "charset": list(self.charset),
            "hidden_dim": self.hidden_dim,
            "keras_version": self.keras_version,
            "keras_backend": self.keras_backend,
            "layer_order": [name for name, _ in _LAYER_SEQUENCE],
            "weight_shapes": {
                layer: {name: list(value.shape) for name, value in tensors.items()}
                for layer, tensors in self.weights.items()
            },
            "weight_dtypes": {
                layer: {name: str(value.dtype) for name, value in tensors.items()}
                for layer, tensors in self.weights.items()
            },
        }


@dataclass(frozen=True)
class ChemicalVaeEncoderRuntime:
    """Architecture metadata sufficient to load a verified converted state.

    Unlike :class:`ChemicalVaeEncoderAsset`, this object deliberately contains
    no source HDF5 weights.  It lets ordinary inference load only the checked
    PyTorch state dict plus its conversion manifest; HDF5/h5py is a conversion
    and audit dependency, not a descriptor runtime dependency.
    """

    source_sha256: str
    variant: str
    max_len: int
    padding: str
    charset: Tuple[str, ...]
    hidden_dim: int
    layer_configs: Mapping[str, Mapping[str, Any]]


def _runtime_from_manifest_fragment(fragment: Mapping[str, Any]) -> ChemicalVaeEncoderRuntime:
    """Validate the immutable v5 source fragment embedded in a state payload."""

    source_sha256 = fragment.get("source_encoder_sha256")
    charset_raw = fragment.get("charset")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ChemicalVaeAssetError("converted encoder payload has an invalid source_encoder_sha256")
    variant = fragment.get("variant")
    if not isinstance(variant, str) or variant not in _VERIFIED_ENCODER_SOURCES:
        raise ChemicalVaeAssetError(
            "converted encoder only supports separately verified zinc or zinc_properties variants, "
            f"got {variant!r}"
        )
    if source_sha256 != _VERIFIED_ENCODER_SOURCES[variant]:
        raise ChemicalVaeAssetError(
            f"converted encoder source hash is not the verified {variant} asset"
        )
    if fragment.get("max_len") != 120 or fragment.get("padding") != "right":
        raise ChemicalVaeAssetError("converted encoder MAX_LEN/padding differ from the verified Chemical VAE contract")
    if not isinstance(charset_raw, list) or len(charset_raw) != 35 or not all(
        isinstance(value, str) and len(value) == 1 for value in charset_raw
    ):
        raise ChemicalVaeAssetError("converted encoder has an invalid verified Chemical VAE charset")
    charset = tuple(charset_raw)
    if len(set(charset)) != len(charset) or " " not in charset:
        raise ChemicalVaeAssetError("converted encoder charset must be unique and include the space padding token")
    if fragment.get("hidden_dim") != 196:
        raise ChemicalVaeAssetError("converted encoder hidden_dim differs from the verified Chemical VAE contract")
    if fragment.get("layer_order") != [name for name, _ in _LAYER_SEQUENCE]:
        raise ChemicalVaeAssetError("converted encoder layer order differs from the verified Chemical VAE contract")
    expected_shapes = {
        "encoder_conv0": {"kernel": [9, 35, 9], "bias": [9]},
        "encoder_conv1": {"kernel": [9, 9, 9], "bias": [9]},
        "encoder_conv2": {"kernel": [11, 9, 10], "bias": [10]},
        "encoder_norm0": {name: [9] for name in ("gamma", "beta", "moving_mean", "moving_variance")},
        "encoder_norm1": {name: [9] for name in ("gamma", "beta", "moving_mean", "moving_variance")},
        "encoder_norm2": {name: [10] for name in ("gamma", "beta", "moving_mean", "moving_variance")},
        "encoder_dense0": {"kernel": [940, 196], "bias": [196]},
        "encoder_dense0_norm": {name: [196] for name in ("gamma", "beta", "moving_mean", "moving_variance")},
        "z_mean_sample": {"kernel": [196, 196], "bias": [196]},
    }
    if fragment.get("weight_shapes") != expected_shapes:
        raise ChemicalVaeAssetError("converted encoder weight shapes differ from the verified Chemical VAE contract")
    return ChemicalVaeEncoderRuntime(
        source_sha256=source_sha256,
        variant=variant,
        max_len=120,
        padding="right",
        charset=charset,
        hidden_dim=196,
        layer_configs=_RUNTIME_LAYER_CONFIGS,
    )


def _layer_map(model_config: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    config = model_config.get("config")
    if not isinstance(config, Mapping):
        raise ChemicalVaeAssetError("Keras model_config lacks mapping config")
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ChemicalVaeAssetError("Keras model_config lacks layers")
    observed = tuple((layer.get("name"), layer.get("class_name")) for layer in layers)
    if observed != _LAYER_SEQUENCE:
        raise ChemicalVaeAssetError(
            "unsupported encoder layer graph; expected "
            f"{_LAYER_SEQUENCE!r}, received {observed!r}"
        )
    output_layers = config.get("output_layers")
    if output_layers != [["z_mean_sample", 0, 0], ["encoder_dense0_norm", 0, 0]]:
        raise ChemicalVaeAssetError(f"unexpected encoder outputs: {output_layers!r}")
    return {str(layer["name"]): dict(layer.get("config", {})) for layer in layers}


def _read_tensor(group: Any, name: str, expected_shape: Tuple[int, ...]) -> np.ndarray:
    if name not in group:
        raise ChemicalVaeAssetError(f"HDF5 weight tensor missing: {group.name}/{name}")
    value = np.asarray(group[name], dtype=np.float32)
    if tuple(value.shape) != expected_shape:
        raise ChemicalVaeAssetError(
            f"HDF5 tensor {group.name}/{name} has shape {tuple(value.shape)!r}; "
            f"expected {expected_shape!r}"
        )
    return value


def _read_hdf5_weights(handle: Any, layer_configs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    if "model_weights" not in handle:
        raise ChemicalVaeAssetError("HDF5 file has no model_weights group")
    root = handle["model_weights"]
    weights: Dict[str, Dict[str, np.ndarray]] = {}
    input_channels = int(layer_configs["input_molecule_smi"]["batch_input_shape"][2])
    previous_channels = input_channels
    for layer_name in _CONVOLUTION_NAMES:
        config = layer_configs[layer_name]
        kernel_size = int(config["kernel_size"][0])
        filters = int(config["filters"])
        group = root[layer_name][layer_name]
        weights[layer_name] = {
            "kernel": _read_tensor(group, "kernel:0", (kernel_size, previous_channels, filters)),
            "bias": _read_tensor(group, "bias:0", (filters,)),
        }
        previous_channels = filters
    flattened_size = 94 * previous_channels
    dense0_group = root["encoder_dense0"]["encoder_dense0"]
    hidden_dim = int(layer_configs["encoder_dense0"]["units"])
    weights["encoder_dense0"] = {
        "kernel": _read_tensor(dense0_group, "kernel:0", (flattened_size, hidden_dim)),
        "bias": _read_tensor(dense0_group, "bias:0", (hidden_dim,)),
    }
    z_mean_group = root["z_mean_sample"]["z_mean_sample"]
    z_mean_dim = int(layer_configs["z_mean_sample"]["units"])
    weights["z_mean_sample"] = {
        "kernel": _read_tensor(z_mean_group, "kernel:0", (hidden_dim, z_mean_dim)),
        "bias": _read_tensor(z_mean_group, "bias:0", (z_mean_dim,)),
    }
    for layer_name in _BATCH_NORM_NAMES:
        channels = hidden_dim if layer_name == "encoder_dense0_norm" else int(
            layer_configs[layer_name.replace("norm", "conv")]["filters"]
        )
        group = root[layer_name][layer_name]
        weights[layer_name] = {
            name: _read_tensor(group, f"{name}:0", (channels,))
            for name in ("gamma", "beta", "moving_mean", "moving_variance")
        }
    return weights


def load_chemical_vae_encoder_asset(
    encoder_path: Path | str,
    *,
    exp_path: Path | str,
    charset_path: Path | str,
    variant: str = "zinc",
) -> ChemicalVaeEncoderAsset:
    """Load and cross-check a supported historical Keras encoder asset.

    The function returns raw source-layout tensors.  It never changes the
    source HDF5 file and performs no decoder/property-model loading.
    """

    source_path = Path(encoder_path).resolve()
    exp_file = Path(exp_path).resolve()
    charset_file = Path(charset_path).resolve()
    for path in (source_path, exp_file, charset_file):
        if not path.is_file():
            raise ChemicalVaeAssetError(f"required Chemical VAE asset is missing: {path}")
    exp = _read_json(exp_file)
    charset_raw = _read_json(charset_file)
    if not isinstance(charset_raw, list) or not charset_raw or not all(
        isinstance(character, str) and len(character) == 1 for character in charset_raw
    ):
        raise ChemicalVaeAssetError("charset must be a non-empty JSON list of one-character strings")
    charset = tuple(charset_raw)
    if len(set(charset)) != len(charset) or " " not in charset:
        raise ChemicalVaeAssetError("charset must contain unique characters including one space padding token")
    max_len = exp.get("MAX_LEN")
    hidden_dim = exp.get("hidden_dim")
    padding = exp.get("PADDING")
    if not isinstance(max_len, int) or max_len <= 0:
        raise ChemicalVaeAssetError(f"invalid exp MAX_LEN: {max_len!r}")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ChemicalVaeAssetError(f"invalid exp hidden_dim: {hidden_dim!r}")
    if padding != "right":
        raise ChemicalVaeAssetError(f"only original right padding is supported; received {padding!r}")

    h5py = _import_h5py()
    with h5py.File(source_path, "r") as handle:
        backend = handle.attrs.get("backend")
        keras_version = handle.attrs.get("keras_version")
        model_config_raw = handle.attrs.get("model_config")
        if isinstance(backend, bytes):
            backend = backend.decode("utf-8")
        if isinstance(keras_version, bytes):
            keras_version = keras_version.decode("utf-8")
        if isinstance(model_config_raw, bytes):
            model_config_raw = model_config_raw.decode("utf-8")
        if backend != "tensorflow" or not isinstance(keras_version, str):
            raise ChemicalVaeAssetError(
                f"expected TensorFlow Keras HDF5 metadata; got backend={backend!r}, keras={keras_version!r}"
            )
        if not isinstance(model_config_raw, str):
            raise ChemicalVaeAssetError("HDF5 model_config is not JSON text")
        layer_configs = _layer_map(json.loads(model_config_raw))
        batch_shape = layer_configs["input_molecule_smi"].get("batch_input_shape")
        expected_input = [None, max_len, len(charset)]
        if batch_shape != expected_input:
            raise ChemicalVaeAssetError(
                f"exp/charset input contract {expected_input!r} conflicts with HDF5 input {batch_shape!r}"
            )
        if int(layer_configs["z_mean_sample"].get("units", -1)) != hidden_dim:
            raise ChemicalVaeAssetError("exp hidden_dim conflicts with HDF5 z_mean_sample units")
        if layer_configs["z_mean_sample"].get("activation") != "linear":
            raise ChemicalVaeAssetError("z_mean_sample must have linear activation")
        weights = _read_hdf5_weights(handle, layer_configs)
    return ChemicalVaeEncoderAsset(
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        variant=variant,
        max_len=max_len,
        padding=padding,
        charset=charset,
        hidden_dim=hidden_dim,
        keras_version=keras_version,
        keras_backend=backend,
        layer_configs=layer_configs,
        weights=weights,
    )


def smiles_to_right_padded_one_hot(smiles: Sequence[str], asset: Any) -> np.ndarray:
    """Apply the original ``smiles_to_hot`` right-padding semantics exactly.

    Every padded position is a one-hot space character, rather than an all-zero
    vector.  This distinction materially changes convolutional encoder output.
    """

    indices = {character: index for index, character in enumerate(asset.charset)}
    padding_index = indices[" "]
    output = np.zeros((len(smiles), asset.max_len, len(asset.charset)), dtype=np.float32)
    output[:, :, padding_index] = 1.0
    for row, value in enumerate(smiles):
        if not isinstance(value, str):
            raise ChemicalVaeAssetError(f"SMILES at row {row} must be str, received {type(value).__name__}")
        if len(value) > asset.max_len:
            raise ChemicalVaeAssetError(
                f"SMILES at row {row} has length {len(value)}, exceeding MAX_LEN={asset.max_len}"
            )
        for column, character in enumerate(value):
            try:
                character_index = indices[character]
            except KeyError as exc:
                raise ChemicalVaeAssetError(
                    f"SMILES at row {row} contains unsupported character {character!r} at offset {column}"
                ) from exc
            output[row, column, padding_index] = 0.0
            output[row, column, character_index] = 1.0
    return output


def _numpy_batch_norm(
    values: np.ndarray, weights: Mapping[str, np.ndarray], epsilon: float
) -> np.ndarray:
    denominator = np.sqrt(weights["moving_variance"] + np.float32(epsilon))
    return (values - weights["moving_mean"]) / denominator * weights["gamma"] + weights["beta"]


def _numpy_valid_conv1d_tanh(values: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch_size, length, channels = values.shape
    kernel_size, kernel_channels, filters = kernel.shape
    if channels != kernel_channels:
        raise ChemicalVaeAssetError("NumPy Conv1D received incompatible channel count")
    output = np.empty((batch_size, length - kernel_size + 1, filters), dtype=np.float32)
    for offset in range(output.shape[1]):
        output[:, offset, :] = np.tensordot(
            values[:, offset : offset + kernel_size, :], kernel, axes=((1, 2), (0, 1))
        ) + bias
    return np.tanh(output).astype(np.float32, copy=False)


def numpy_encoder_forward(
    one_hot: np.ndarray, asset: ChemicalVaeEncoderAsset, *, capture_layers: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Independent channels-last NumPy reference for the audited Keras graph."""

    values = np.asarray(one_hot, dtype=np.float32)
    expected = (asset.max_len, len(asset.charset))
    if values.ndim != 3 or tuple(values.shape[1:]) != expected:
        raise ChemicalVaeAssetError(
            f"NumPy encoder expected (n, {expected[0]}, {expected[1]}), received {tuple(values.shape)!r}"
        )
    layers: Dict[str, np.ndarray] = {"input_molecule_smi": values.copy()} if capture_layers else {}
    current = values
    for index, conv_name in enumerate(_CONVOLUTION_NAMES):
        current = _numpy_valid_conv1d_tanh(
            current, asset.weights[conv_name]["kernel"], asset.weights[conv_name]["bias"]
        )
        if capture_layers:
            layers[conv_name] = current.copy()
        norm_name = _BATCH_NORM_NAMES[index]
        current = _numpy_batch_norm(
            current, asset.weights[norm_name], float(asset.layer_configs[norm_name]["epsilon"])
        ).astype(np.float32, copy=False)
        if capture_layers:
            layers[norm_name] = current.copy()
    current = current.reshape((current.shape[0], -1), order="C")
    if capture_layers:
        layers["flatten_1"] = current.copy()
    current = np.tanh(current @ asset.weights["encoder_dense0"]["kernel"] + asset.weights["encoder_dense0"]["bias"])
    current = current.astype(np.float32, copy=False)
    if capture_layers:
        layers["encoder_dense0"] = current.copy()
        # Inference-mode Keras Dropout is identity.  Persist this layer to make
        # that semantic choice visible in parity evidence.
        layers["dropout_1"] = current.copy()
    current = _numpy_batch_norm(
        current,
        asset.weights["encoder_dense0_norm"],
        float(asset.layer_configs["encoder_dense0_norm"]["epsilon"]),
    ).astype(np.float32, copy=False)
    if capture_layers:
        layers["encoder_dense0_norm"] = current.copy()
    current = current @ asset.weights["z_mean_sample"]["kernel"] + asset.weights["z_mean_sample"]["bias"]
    current = current.astype(np.float32, copy=False)
    if capture_layers:
        layers["z_mean_sample"] = current.copy()
    return current, layers


def _torch_encoder_class(torch: Any) -> Any:
    """Create the candidate module lazily so normal imports do not require torch."""

    nn = torch.nn

    class TorchChemicalVaeEncoder(nn.Module):
        def __init__(self, asset: Any, *, load_source_weights: bool = True) -> None:
            super().__init__()
            self.max_len = asset.max_len
            self.charset_size = len(asset.charset)
            conv_layers = []
            input_channels = self.charset_size
            for conv_name in _CONVOLUTION_NAMES:
                config = asset.layer_configs[conv_name]
                conv_layers.append(
                    nn.Conv1d(
                        input_channels,
                        int(config["filters"]),
                        kernel_size=int(config["kernel_size"][0]),
                        stride=int(config["strides"][0]),
                        padding=0,
                        bias=True,
                    )
                )
                input_channels = int(config["filters"])
            self.convs = nn.ModuleList(conv_layers)
            self.conv_norms = nn.ModuleList(
                [
                    nn.BatchNorm1d(
                        int(asset.layer_configs[conv_name]["filters"]),
                        eps=float(asset.layer_configs[norm_name]["epsilon"]),
                        affine=True,
                        track_running_stats=True,
                    )
                    for conv_name, norm_name in zip(_CONVOLUTION_NAMES, _BATCH_NORM_NAMES[:3])
                ]
            )
            self.dense0 = nn.Linear(94 * input_channels, asset.hidden_dim, bias=True)
            self.dense0_norm = nn.BatchNorm1d(
                asset.hidden_dim,
                eps=float(asset.layer_configs["encoder_dense0_norm"]["epsilon"]),
                affine=True,
                track_running_stats=True,
            )
            self.z_mean = nn.Linear(asset.hidden_dim, asset.hidden_dim, bias=True)
            if load_source_weights:
                self._load_keras_weights(asset, torch)
            self.eval()

        def _copy_batch_norm(self, target: Any, source: Mapping[str, np.ndarray], torch_module: Any) -> None:
            with torch_module.no_grad():
                target.weight.copy_(torch_module.from_numpy(source["gamma"]))
                target.bias.copy_(torch_module.from_numpy(source["beta"]))
                target.running_mean.copy_(torch_module.from_numpy(source["moving_mean"]))
                target.running_var.copy_(torch_module.from_numpy(source["moving_variance"]))

        def _load_keras_weights(self, asset: ChemicalVaeEncoderAsset, torch_module: Any) -> None:
            with torch_module.no_grad():
                for conv, conv_name in zip(self.convs, _CONVOLUTION_NAMES):
                    source = asset.weights[conv_name]
                    conv.weight.copy_(torch_module.from_numpy(np.transpose(source["kernel"], (2, 1, 0))))
                    conv.bias.copy_(torch_module.from_numpy(source["bias"]))
                for norm, norm_name in zip(self.conv_norms, _BATCH_NORM_NAMES[:3]):
                    self._copy_batch_norm(norm, asset.weights[norm_name], torch_module)
                self.dense0.weight.copy_(torch_module.from_numpy(asset.weights["encoder_dense0"]["kernel"].T))
                self.dense0.bias.copy_(torch_module.from_numpy(asset.weights["encoder_dense0"]["bias"]))
                self._copy_batch_norm(self.dense0_norm, asset.weights["encoder_dense0_norm"], torch_module)
                self.z_mean.weight.copy_(torch_module.from_numpy(asset.weights["z_mean_sample"]["kernel"].T))
                self.z_mean.bias.copy_(torch_module.from_numpy(asset.weights["z_mean_sample"]["bias"]))

        def forward(self, one_hot: Any, capture_layers: bool = False) -> Any:
            if one_hot.ndim != 3 or tuple(one_hot.shape[1:]) != (self.max_len, self.charset_size):
                raise ChemicalVaeAssetError(
                    "PyTorch encoder expected "
                    f"(n, {self.max_len}, {self.charset_size}), received {tuple(one_hot.shape)!r}"
                )
            layers: MutableMapping[str, Any] = {}
            if capture_layers:
                layers["input_molecule_smi"] = one_hot
            current = one_hot.transpose(1, 2)
            for conv, norm, conv_name, norm_name in zip(
                self.convs, self.conv_norms, _CONVOLUTION_NAMES, _BATCH_NORM_NAMES[:3]
            ):
                current = torch.tanh(conv(current))
                if capture_layers:
                    layers[conv_name] = current.transpose(1, 2)
                current = norm(current)
                if capture_layers:
                    layers[norm_name] = current.transpose(1, 2)
            current = current.transpose(1, 2).contiguous().view(current.shape[0], -1)
            if capture_layers:
                layers["flatten_1"] = current
            current = torch.tanh(self.dense0(current))
            if capture_layers:
                layers["encoder_dense0"] = current
                layers["dropout_1"] = current
            current = self.dense0_norm(current)
            if capture_layers:
                layers["encoder_dense0_norm"] = current
            current = self.z_mean(current)
            if capture_layers:
                layers["z_mean_sample"] = current
                return current, dict(layers)
            return current

    return TorchChemicalVaeEncoder


def build_torch_encoder(asset: Any, *, load_source_weights: bool = True) -> Any:
    """Build the candidate PyTorch model from independently read raw HDF5 tensors."""

    torch = _import_torch()
    return _torch_encoder_class(torch)(asset, load_source_weights=load_source_weights)


def torch_encoder_forward(
    one_hot: np.ndarray, asset: ChemicalVaeEncoderAsset, *, capture_layers: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run a fresh candidate model directly from source HDF5 tensors on CPU."""

    torch = _import_torch()
    model = build_torch_encoder(asset).cpu().eval()
    tensor = torch.from_numpy(np.asarray(one_hot, dtype=np.float32)).cpu()
    with torch.no_grad():
        if capture_layers:
            output, layers = model(tensor, capture_layers=True)
            return (
                output.detach().cpu().numpy().astype(np.float32, copy=False),
                {
                    name: value.detach().cpu().numpy().astype(np.float32, copy=False)
                    for name, value in layers.items()
                },
            )
        output = model(tensor)
    return output.detach().cpu().numpy().astype(np.float32, copy=False), {}


def converted_payload(asset: ChemicalVaeEncoderAsset, *, converter_version: str) -> Dict[str, Any]:
    """Create the persisted candidate payload without writing it to disk."""

    model = build_torch_encoder(asset).cpu().eval()
    return {
        "schema_version": "chemical_vae_torch_encoder/v1",
        "converter_version": converter_version,
        "asset": asset.manifest_fragment(),
        "state_dict": model.state_dict(),
    }


def load_converted_torch_encoder(
    converted_path: Path | str, asset: ChemicalVaeEncoderAsset
) -> Any:
    """Load a converted state dict and reject an asset-identity mismatch."""

    torch = _import_torch()
    path = Path(converted_path)
    if not path.is_file():
        raise ChemicalVaeAssetError(f"converted encoder does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before the optional weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ChemicalVaeAssetError("converted encoder payload is not a mapping")
    expected = asset.source_sha256
    observed = payload.get("asset", {}).get("source_encoder_sha256") if isinstance(payload.get("asset"), Mapping) else None
    if observed != expected:
        raise ChemicalVaeAssetError(
            f"converted source identity mismatch: expected {expected}, received {observed}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ChemicalVaeAssetError("converted encoder payload has no state_dict")
    # Construct only the validated architecture; candidate values must come
    # exclusively from the persisted state dict, never from source HDF5 tensors.
    model = build_torch_encoder(asset, load_source_weights=False).cpu().eval()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ChemicalVaeAssetError(f"converted state_dict does not match audited architecture: {exc}") from exc
    return model


def load_converted_torch_encoder_runtime(
    converted_path: Path | str,
    *,
    expected_source_sha256: str | None = None,
) -> Tuple[Any, ChemicalVaeEncoderRuntime]:
    """Load a checked converted state without reopening source HDF5 weights.

    The payload's source-manifest fragment is validated against the frozen
    per-variant encoder identity and architecture before a strict ``state_dict`` load.  Thus ordinary
    descriptor inference does not import h5py or access the reference project.
    """

    torch = _import_torch()
    path = Path(converted_path)
    if not path.is_file():
        raise ChemicalVaeAssetError(f"converted encoder does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before the optional weights_only argument.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ChemicalVaeAssetError("converted encoder payload is not a mapping")
    if payload.get("schema_version") != "chemical_vae_torch_encoder/v1":
        raise ChemicalVaeAssetError(f"unsupported converted encoder schema: {payload.get('schema_version')!r}")
    fragment = payload.get("asset")
    if not isinstance(fragment, Mapping):
        raise ChemicalVaeAssetError("converted encoder payload has no asset manifest fragment")
    runtime = _runtime_from_manifest_fragment(fragment)
    if expected_source_sha256 is not None and runtime.source_sha256 != expected_source_sha256:
        raise ChemicalVaeAssetError(
            "converted source identity mismatch: "
            f"expected {expected_source_sha256}, received {runtime.source_sha256}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ChemicalVaeAssetError("converted encoder payload has no state_dict")
    model = build_torch_encoder(runtime, load_source_weights=False).cpu().eval()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ChemicalVaeAssetError(f"converted state_dict does not match audited architecture: {exc}") from exc
    return model, runtime


def converted_torch_encoder_forward(
    one_hot: np.ndarray, converted_path: Path | str, asset: ChemicalVaeEncoderAsset, *, capture_layers: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run the candidate that was actually persisted by the conversion tool."""

    torch = _import_torch()
    model = load_converted_torch_encoder(converted_path, asset)
    tensor = torch.from_numpy(np.asarray(one_hot, dtype=np.float32)).cpu()
    with torch.no_grad():
        if capture_layers:
            output, layers = model(tensor, capture_layers=True)
            return (
                output.detach().cpu().numpy().astype(np.float32, copy=False),
                {
                    name: value.detach().cpu().numpy().astype(np.float32, copy=False)
                    for name, value in layers.items()
                },
            )
        output = model(tensor)
    return output.detach().cpu().numpy().astype(np.float32, copy=False), {}

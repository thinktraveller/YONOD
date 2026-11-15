"""Offline MolMetaLM asset admission, without importing Torch or Transformers.

The built-in pins identify the authors' exact upstream checkpoint; they were
checked against the fixed HF revision below, including its LFS SHA-256 for the
binary. A manifest found beside a checkpoint is never an authority. Callers
supplying ``asset_sha256`` must obtain those pins through a separate trusted
review, not by hashing an arbitrary downloaded directory and trusting it.

This validates inputs to the loader, not an entire vulnerable dependency or a
host on which another process can rewrite the application/asset files. Deploy
approved assets read-only for the lifetime of the descriptor.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional


UPSTREAM_REPOSITORY = "wudejian789/MolMetaLM-base"
UPSTREAM_REVISION = "cc29077bc0384ee59d6a17fc91c1f0c9a764cd96"
# Author's README and InferenceCollateFunc at CSUBioGroup/MolMetaLM commit
# 9b956408483c385949ddecaef0452fcb999de896 use ' '.join(list(smi)). The
# approved fast tokenizer has no postprocessor/special-token insertion.
INPUT_PREPROCESSING_POLICY = "author-char-whitespace-fast-truncate512-reject-unk/v1"
# https://huggingface.co/api/models/wudejian789/MolMetaLM-base?blobs=true
# Small files checked at /resolve/<UPSTREAM_REVISION>/<filename>; model binary
# checked against the revision's published LFS oid, without downloading it.
DEFAULT_ASSET_SHA256 = MappingProxyType({
    "config.json": "eba91add65e0ce14ecd6038a90aa6ddd57c60c1473c8eb7a2a7016f5daf4417b",
    "tokenizer_config.json": "8237146884cb46264178141ed1aeb5f3089e0f531579c32c9e26e0c0f5033cc6",
    "tokenizer.json": "bbb874db34e1bb65470db3cfc01e242e23d09c6206d97e4e892360edbf7591b4",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "added_tokens.json": "aa86d81abbb1c470b24e77f496b65b71f10141bbed0ad12448d6ecf9bfe7bdfc",
    "vocab.txt": "71291f5ca018cc808067ab0c14af0de19a1b07b723ce1d8d19120bcc605c7141",
    "generation_config.json": "906e155b9c304db7c0a35799c39795e7d768bbc3a22fb0edf007c9207e6de76e",
    "model.safetensors": "40dcda5fb361c0533b07393d71d940be41792a1ff96ebeed89bf4ba2ef503d24",
})

_CONFIG_FIELDS = frozenset({
    "_name_or_path", "architectures", "attention_bias", "attention_dropout",
    "bos_token_id", "decoder_start_token_id", "eos_token_id", "forced_eos_token_id",
    "hidden_act", "hidden_size", "initializer_range", "intermediate_size",
    "mask_token_id", "max_position_embeddings", "model_type", "num_attention_heads",
    "num_hidden_layers", "num_key_value_heads", "pad_token_id", "position_encoding_2d",
    "pretraining_tp", "rms_norm_eps", "rope_scaling", "rope_theta", "tie_word_embeddings",
    "torch_dtype", "transformers_version", "use_cache", "vocab_size",
})
_TOKENIZER_FIELDS = frozenset({
    "added_tokens_decoder", "clean_up_tokenization_spaces", "cls_token", "do_lower_case",
    "mask_token", "model_max_length", "pad_token", "sep_token", "strip_accents",
    "tokenize_chinese_chars", "tokenizer_class", "unk_token",
})
_GENERATION_FIELDS = frozenset({
    "bos_token_id", "decoder_start_token_id", "eos_token_id", "forced_eos_token_id",
    "length_penalty", "mask_token_id", "no_repeat_ngram_size", "num_beams", "pad_token_id",
    "transformers_version",
})
_TOKEN_FIELDS = frozenset({"cls_token", "mask_token", "pad_token", "sep_token", "unk_token"})
_FORBIDDEN_FIELDS = frozenset({
    "auto_map", "configuration_files", "trust_remote_code", "code_revision",
    "custom_pipelines", "chat_template", "chat_templates", "attn_implementation",
    "experts_implementation",
})
_MAX_METADATA_BYTES = 4 * 1024 * 1024


class MolMetaLMAssetError(ValueError):
    """The checkpoint did not satisfy the approved local loading policy."""


def character_spaced_smiles(smiles: str) -> str:
    """Apply the author character policy; vocabulary coverage is checked later.

    This deliberately splits Cl, Br, brackets and ring numbers into individual
    characters. A generic atom/SMILES regex would change the trained input.
    """
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("MolMetaLM requires a nonempty SMILES string")
    return " ".join(list(smiles))


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MolMetaLMAssetError("Duplicate JSON key in MolMetaLM metadata: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise MolMetaLMAssetError("Non-finite JSON constant in MolMetaLM metadata: " + value)


def _read_json(path: Path) -> Dict[str, Any]:
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise MolMetaLMAssetError("MolMetaLM metadata is too large: " + path.name)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MolMetaLMAssetError("Invalid MolMetaLM metadata: " + path.name) from exc
    if not isinstance(value, dict):
        raise MolMetaLMAssetError("MolMetaLM metadata must be a JSON object: " + path.name)
    _reject_dynamic_fields(value)
    return value


def _reject_dynamic_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if (
                lowered in _FORBIDDEN_FIELDS or "kernel" in lowered
                or lowered.startswith("_attn_") or lowered.startswith("_experts_")
            ):
                raise MolMetaLMAssetError("Unapproved dynamic MolMetaLM field: " + key)
            _reject_dynamic_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_dynamic_fields(item)


def _allowed_fields(value: Dict[str, Any], allowed: frozenset, name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise MolMetaLMAssetError("Unapproved fields in " + name + ": " + ", ".join(sorted(unknown)))


def _validate_metadata(root: Path) -> None:
    config = _read_json(root / "config.json")
    _allowed_fields(config, _CONFIG_FIELDS, "config.json")
    if (
        config.get("model_type") != "llama"
        or config.get("architectures") != ["LlamaForCausalLM"]
        or type(config.get("hidden_size")) is not int or config["hidden_size"] != 768
        or config.get("torch_dtype") != "float32" or config.get("rope_scaling") is not None
    ):
        raise MolMetaLMAssetError("MolMetaLM requires the approved native Llama 768/float32 profile")
    # This profile contains no nested model configuration. In particular a
    # trusted pin cannot silently expand admission to a LightGlue submodel.
    if any(
        isinstance(value, dict) or (isinstance(value, list) and key != "architectures")
        for key, value in config.items()
    ):
        raise MolMetaLMAssetError("Nested model configuration is not supported for MolMetaLM")
    tokenizer = _read_json(root / "tokenizer_config.json")
    _allowed_fields(tokenizer, _TOKENIZER_FIELDS, "tokenizer_config.json")
    if tokenizer.get("tokenizer_class") != "BertTokenizer":
        raise MolMetaLMAssetError("MolMetaLM requires its approved BertTokenizer")
    _allowed_fields(_read_json(root / "generation_config.json"), _GENERATION_FIELDS, "generation_config.json")
    _allowed_fields(_read_json(root / "special_tokens_map.json"), _TOKEN_FIELDS, "special_tokens_map.json")
    _read_json(root / "added_tokens.json")
    fast = _read_json(root / "tokenizer.json")
    if not isinstance(fast.get("model"), dict) or fast["model"].get("type") != "WordPiece":
        raise MolMetaLMAssetError("MolMetaLM requires its approved WordPiece tokenizer model")


def _approved_pins(asset_sha256: Optional[Mapping]) -> Dict[str, str]:
    pins = DEFAULT_ASSET_SHA256 if asset_sha256 is None else asset_sha256
    if not isinstance(pins, Mapping) or set(pins) != set(DEFAULT_ASSET_SHA256):
        raise MolMetaLMAssetError("Explicit MolMetaLM pins must name exactly the eight approved asset files")
    result = {}
    for name, digest in pins.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise MolMetaLMAssetError("Invalid SHA-256 pin for MolMetaLM asset: " + name)
        result[name] = digest.lower()
    return result


def validated_asset_identity(
    weight_path: str, asset_sha256: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Validate local inputs and return a stable, JSON-serializable identity.

    ``asset_sha256`` is an explicit caller trust decision, never a directory
    manifest or an automatic trust-on-first-use fallback. It still cannot
    enable other architectures, remote code, pickle weights or nested models.
    No model code is imported or executed by this function.
    """
    pins = _approved_pins(asset_sha256)
    if "://" in str(weight_path):
        raise MolMetaLMAssetError("MolMetaLM accepts a local asset directory only")
    root = Path(weight_path).expanduser()
    try:
        if root.is_symlink():
            raise MolMetaLMAssetError("MolMetaLM asset directory must not be a symlink")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise MolMetaLMAssetError("MolMetaLM asset path must be a directory")
        allowed = set(pins) | {"README.md", ".gitattributes", ".cache"}
        for entry in root.iterdir():
            if entry.name not in allowed or entry.is_symlink():
                raise MolMetaLMAssetError("Unapproved MolMetaLM asset entry: " + entry.name)
            if entry.is_dir() and entry.name != ".cache":
                raise MolMetaLMAssetError("Unexpected MolMetaLM asset directory: " + entry.name)
            if entry.name == ".cache" and not entry.is_dir():
                raise MolMetaLMAssetError("MolMetaLM .cache entry must be a directory")
        for name, expected in pins.items():
            path = root / name
            if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                raise MolMetaLMAssetError("Missing or unsafe MolMetaLM asset: " + name)
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise MolMetaLMAssetError("MolMetaLM asset changed during validation: " + name)
            if digest.hexdigest() != expected:
                raise MolMetaLMAssetError("SHA-256 mismatch for MolMetaLM asset: " + name)
        _validate_metadata(root)
    except (OSError, RuntimeError, RecursionError) as exc:
        raise MolMetaLMAssetError("Cannot validate local MolMetaLM assets") from exc
    policy = {"kind": "explicit-caller-pins"}
    if asset_sha256 is None:
        policy = {"kind": "pinned-upstream", "repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION}
    return {
        "schema": "yonod.molmetalm-assets/v1", "source_policy": policy,
        "files": [{"path": name, "sha256": pins[name]} for name in sorted(pins)],
    }

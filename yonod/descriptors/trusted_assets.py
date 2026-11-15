"""Independent admission pins for pickle-based pretrained descriptor assets.

Only standard-library modules are imported. Torch <2.10 weights_only loading
is not a sandbox (CVE-2026-24747): the application accepts only these exact
independently obtained checkpoint bytes. A neighbouring manifest is not a
trust authority. This admission policy does not patch Torch itself.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import re


ASSET_REPOSITORY = "thinktraveller/YONOD-weights"
ASSET_REVISION = "861179c95ae30d0f5a17ce63a755ebe42d5db250"
FISD_CHECKPOINT_PINS = {
    "qm_9_2in1_model.pth": "28905d23e313322112c78e778079c55aec272c28f8acdc649bf3488d2e69bcd7",
    "qm_9_cos_model.pth": "062fca53630b028a8d973f5dd54abe604a71e169f26c6896e74f40a57b1c18be",
    "qm_9_mse_model.pth": "65131ef4520ce69f848cea3a5ef7d3f16ff002eee047b8a1dfea7348daa2a19b",
}
CHEMICAL_VAE_PINS = {
    "zinc": {
        "path": "chemical_vae/zinc-37e96cd3bc8f9680/v5",
        "manifest_sha256": "bf93a1e2321fa2c38f537bad1974660f11435651e3e3dbe8e0d41a6df9563ac2",
        "state_sha256": "1bf5954e66cff67895e241bb7dcfb4d719874e8a1cf9d5c6cd755a7c81d35001",
        "source_encoder_sha256": "37e96cd3bc8f9680d3c2aa4d294a39e78f00a0decc22501bde23d5a464ac314a",
    },
    "zinc_properties": {
        "path": "chemical_vae/zinc_properties-9f923d03c5ce558d/v5",
        "manifest_sha256": "55eda62cfee6e6f2dedcf6830e31282c9a78fb4cd3b441eb99efe498a4c63de8",
        "state_sha256": "794d80d1b43285c754987f0f9829df7ebe5f6a7d41d973130d771a994f28508d",
        "source_encoder_sha256": "9f923d03c5ce558d1f0cb52d2a2f4e28f080e59ac5a8c81a4365d7b63ba9d1e2",
    },
}


class TrustedAssetError(ValueError):
    """The file was not admitted by an independent checkpoint digest."""


def read_pinned_asset_bytes(path: Path | str, expected_sha256: str) -> bytes:
    """Return the exact hashed snapshot; never reopen a path for deserialization.

    The expected digest is a caller's independent trust decision. It must not
    be copied from an untrusted manifest alongside the file being admitted.
    """
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise TrustedAssetError("An independently trusted lowercase SHA256 is required")
    source = Path(path).expanduser()
    try:
        if not source.is_file():
            raise TrustedAssetError(f"Trusted pretrained asset is missing or not a file: {source}")
        data = source.read_bytes()
    except OSError as exc:
        raise TrustedAssetError(f"Cannot read trusted pretrained asset: {source}") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise TrustedAssetError(f"Pretrained asset SHA256 is not independently approved: {source}")
    return data


def validated_fisd_asset_identity(model_dir: Path | str) -> dict:
    """Validate all three checkpoints without importing Torch or model code."""
    folder = Path(model_dir).expanduser()
    for name, digest in FISD_CHECKPOINT_PINS.items():
        read_pinned_asset_bytes(folder / name, digest)
    return {
        "schema": "yonod.fisd-assets/v1",
        "repository": ASSET_REPOSITORY, "revision": ASSET_REVISION,
        "source_policy": "independent-fixed-sha256-before-torch-load/v1",
        "files": dict(FISD_CHECKPOINT_PINS),
    }


def validated_chemical_vae_asset_identity(model_manifest: Path | str) -> dict:
    """Admit the frozen manifest and its independently pinned adjacent state."""
    path = Path(model_manifest).expanduser()
    try:
        if not path.is_file():
            raise TrustedAssetError(f"Trusted Chemical VAE manifest is missing or not a file: {path}")
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise TrustedAssetError(f"Trusted Chemical VAE manifest is missing or unreadable: {path}") from exc
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    for variant, pins in CHEMICAL_VAE_PINS.items():
        if manifest_sha == pins["manifest_sha256"]:
            read_pinned_asset_bytes(path.parent / "encoder_state_dict.pt", pins["state_sha256"])
            return {
                "schema": "yonod.chemical-vae-assets/v1", "variant": variant,
                "repository": ASSET_REPOSITORY, "revision": ASSET_REVISION,
                "source_policy": "independent-fixed-sha256-before-torch-load/v1",
                "conversion_manifest_sha256": manifest_sha,
                "converted_state_sha256": pins["state_sha256"],
                "source_encoder_sha256": pins["source_encoder_sha256"],
            }
    raise TrustedAssetError(f"Chemical VAE manifest SHA256 is not independently approved: {path}")


def chemical_vae_checkpoint_buffer(
    path: Path | str, *, trusted_state_sha256: str | None = None,
    expected_source_sha256: str | None = None,
) -> BytesIO:
    """Admit official state bytes, or an explicitly trusted migration output.

    A migration caller may supply the digest of output it created/audited in
    its own trusted process. Omission never means trust-on-first-use and the
    expected source-HDF5 digest is not sufficient to admit serialized state.
    """
    source = Path(path).expanduser()
    try:
        if not source.is_file():
            raise TrustedAssetError(f"Trusted Chemical VAE state is missing or not a file: {source}")
        data = (read_pinned_asset_bytes(source, trusted_state_sha256)
                if trusted_state_sha256 is not None else source.read_bytes())
    except OSError as exc:
        raise TrustedAssetError(f"Trusted Chemical VAE state is missing or unreadable: {source}") from exc
    observed = hashlib.sha256(data).hexdigest()
    approved = next((pins for pins in CHEMICAL_VAE_PINS.values() if pins["state_sha256"] == observed), None)
    if approved is None and trusted_state_sha256 is None:
        raise TrustedAssetError(
            "Chemical VAE state SHA256 is not independently approved; migration callers must explicitly "
            "supply trusted_state_sha256 obtained from their own trusted conversion/audit, not a checkpoint manifest"
        )
    if approved is not None and expected_source_sha256 is not None and approved["source_encoder_sha256"] != expected_source_sha256:
        raise TrustedAssetError("Chemical VAE independently pinned checkpoint has a different source encoder identity")
    return BytesIO(data)

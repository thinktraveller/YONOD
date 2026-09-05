"""Public feature registry shared by ordinary YONOD entry points.

The user-facing "descriptor library" contains two deliberately different
lifecycles: stateless molecular descriptors can be precomputed, while a
fold-transform such as one-hot encoding must be fitted from every training
fold. Keeping that distinction in one registry prevents a configuration from
accidentally sending OHE through the descriptor-artifact cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class FeatureRegistryError(ValueError):
    """Raised for an invalid public feature declaration."""


@dataclass(frozen=True)
class FeatureProvider:
    """Immutable public metadata for one selectable feature provider."""

    feature_name: str
    lifecycle: str
    input_type: str
    modes: tuple[str, ...]
    parameter_schema: Mapping[str, str]
    hint: str


@dataclass(frozen=True)
class FeatureSpec:
    """A normalized, hashable declaration for one feature candidate."""

    id: str
    descriptor: str
    lifecycle: str
    columns: tuple[str, ...]
    mode: str
    params: Mapping[str, Any]
    extra_reactants: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "descriptor": self.descriptor,
            "lifecycle": self.lifecycle,
            "columns": list(self.columns),
            "mode": self.mode,
            "params": dict(self.params),
        }
        if self.extra_reactants:
            result["extra_reactants"] = list(self.extra_reactants)
        return result

    @property
    def schema_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


_STATIC = "static_descriptor"
_FOLD = "fold_transform"

_PROVIDERS: dict[str, FeatureProvider] = {
    "morgan": FeatureProvider("morgan", _STATIC, "smiles", ("concat",), {}, "二值 ECFP4 Morgan 指纹"),
    "mfp": FeatureProvider(
        "mfp", _STATIC, "smiles", ("concat",),
        {"radius": "int >= 0", "fp_size": "int >= 8", "profile": "standard | vjethbkm"},
        "逐列拼接的 Morgan count fingerprint（不是 morgan 的别名）",
    ),
    "maccs": FeatureProvider("maccs", _STATIC, "smiles", ("concat",), {}, "MACCS keys"),
    "fisd": FeatureProvider("fisd", _STATIC, "smiles", ("concat",), {}, "FISD 嵌入"),
    "molmetalm": FeatureProvider("molmetalm", _STATIC, "smiles", ("concat",), {}, "MolMetaLM 嵌入"),
    "maf": FeatureProvider("maf", _STATIC, "smiles", ("sum",), {}, "多分子加和指纹"),
    "rdkit2d": FeatureProvider("rdkit2d", _STATIC, "smiles", ("concat",), {}, "RDKit 2D 描述符"),
    "drfp": FeatureProvider("drfp", _STATIC, "reaction", ("reaction",), {}, "反应差分指纹"),
    "ohe": FeatureProvider(
        "ohe", _FOLD, "categorical_columns", ("concat",),
        {"missing_policy": "as_category | zero_block | error", "dtype": "float32 | float64"},
        "仅在每个 CV 训练折拟合的 one-hot 类别编码",
    ),
}


def get_feature_provider(name: str) -> FeatureProvider:
    """Return public metadata or raise an actionable error for unknown names."""
    normalised = str(name).strip().lower()
    try:
        return _PROVIDERS[normalised]
    except KeyError as exc:
        choices = ", ".join(f"{item.feature_name} ({item.lifecycle})" for item in _PROVIDERS.values())
        raise FeatureRegistryError(f"未知特征 {name!r}。可用特征：{choices}") from exc


def available_feature_names(*, include_fold_transforms: bool = True) -> tuple[str, ...]:
    """Return stable feature names in user-interface order."""
    return tuple(
        name for name, provider in _PROVIDERS.items()
        if include_fold_transforms or provider.lifecycle == _STATIC
    )


def _string_list(value: Any, field: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        if required:
            raise FeatureRegistryError(f"{field} 不能为空")
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FeatureRegistryError(f"{field} 必须是字符串列表")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise FeatureRegistryError(f"{field} 不能包含空列名")
    if len(set(result)) != len(result):
        raise FeatureRegistryError(f"{field} 不能重复")
    if required and not result:
        raise FeatureRegistryError(f"{field} 不能为空")
    return result


def _normalise_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if name == "mfp":
        unknown = set(result).difference({"radius", "fp_size", "profile"})
        if unknown:
            raise FeatureRegistryError(f"mfp.params 包含未知字段：{sorted(unknown)}")
        radius = int(result.get("radius", 3))
        fp_size = int(result.get("fp_size", 1024))
        profile = str(result.get("profile", "standard"))
        if radius < 0 or fp_size < 8:
            raise FeatureRegistryError("mfp.radius 必须 >= 0 且 mfp.fp_size 必须 >= 8")
        if profile not in {"standard", "vjethbkm"}:
            raise FeatureRegistryError("mfp.profile 仅支持 standard 或 vjethbkm")
        return {"radius": radius, "fp_size": fp_size, "profile": profile}
    if name == "ohe":
        unknown = set(result).difference({"missing_policy", "dtype", "handle_unknown"})
        if unknown:
            raise FeatureRegistryError(f"ohe.params 包含未知字段：{sorted(unknown)}")
        policy = str(result.get("missing_policy", "as_category"))
        dtype = str(result.get("dtype", "float32"))
        handle_unknown = str(result.get("handle_unknown", "ignore"))
        if policy not in {"as_category", "zero_block", "error"}:
            raise FeatureRegistryError("ohe.missing_policy 仅支持 as_category、zero_block 或 error")
        if dtype not in {"float32", "float64"}:
            raise FeatureRegistryError("ohe.dtype 仅支持 float32 或 float64")
        if handle_unknown != "ignore":
            raise FeatureRegistryError("ohe.handle_unknown 当前固定为 ignore")
        return {"missing_policy": policy, "dtype": dtype, "handle_unknown": handle_unknown}
    if result:
        raise FeatureRegistryError(f"{name}.params 暂不支持参数：{sorted(result)}")
    return {}


def normalise_feature_specs(
    declarations: Sequence[Mapping[str, Any]],
    *,
    label_col: str | None = None,
) -> list[FeatureSpec]:
    """Validate old/new JSON declarations and produce stable feature specs."""
    if not isinstance(declarations, Sequence) or isinstance(declarations, (str, bytes)):
        raise FeatureRegistryError("descriptors 必须是对象列表")
    specs: list[FeatureSpec] = []
    ids: set[str] = set()
    allowed = {"id", "descriptor", "lifecycle", "columns", "mode", "params", "extra_reactants"}
    for index, raw in enumerate(declarations):
        if not isinstance(raw, Mapping):
            raise FeatureRegistryError(f"descriptors[{index}] 必须是对象")
        unknown = set(raw).difference(allowed)
        if unknown:
            raise FeatureRegistryError(f"descriptors[{index}] 包含未知字段：{sorted(unknown)}")
        if "descriptor" not in raw:
            raise FeatureRegistryError(f"descriptors[{index}].descriptor 不能为空")
        provider = get_feature_provider(str(raw["descriptor"]))
        declared_lifecycle = raw.get("lifecycle")
        if declared_lifecycle is not None and str(declared_lifecycle) != provider.lifecycle:
            raise FeatureRegistryError(
                f"{provider.feature_name}.lifecycle 由注册表固定为 {provider.lifecycle}"
            )
        feature_id = str(raw.get("id", provider.feature_name)).strip()
        if not feature_id:
            raise FeatureRegistryError(f"descriptors[{index}].id 不能为空")
        if feature_id in ids:
            raise FeatureRegistryError(f"feature id 重复：{feature_id!r}；请为参数化候选项提供不同 id")
        ids.add(feature_id)
        mode = str(raw.get("mode", provider.modes[0]))
        if mode not in provider.modes:
            raise FeatureRegistryError(f"{provider.feature_name}.mode 仅支持：{', '.join(provider.modes)}")
        columns = _string_list(raw.get("columns"), f"descriptors[{index}].columns", required=provider.lifecycle == _FOLD)
        if label_col is not None and label_col in columns:
            raise FeatureRegistryError("OHE/描述符输入列不能包含标签列")
        params_raw = raw.get("params", {})
        if not isinstance(params_raw, Mapping):
            raise FeatureRegistryError(f"descriptors[{index}].params 必须是对象")
        params = _normalise_params(provider.feature_name, params_raw)
        extra = _string_list(raw.get("extra_reactants"), f"descriptors[{index}].extra_reactants")
        specs.append(FeatureSpec(feature_id, provider.feature_name, provider.lifecycle, columns, mode, params, extra))
    return specs

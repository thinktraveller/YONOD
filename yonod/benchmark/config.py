"""Benchmark 配置、数据版本与运行清单契约。

本模块刻意不训练模型。它负责在任何特征或模型计算之前固化一次
benchmark 的数据、列角色和评价配置，使后续 split manifest、预测分片和
统计结果能够指向同一份不可歧义的运行元数据。
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml

from ..config.contracts import ConfigContractError, resolve_config_path
from ..config.loader import ConfigLoadError, load_run_config
from ..model_factory import ModelConfigurationError, resolve_model_config
from ..features.numeric_conditions import NumericConditionsError, normalise_numeric_contract, parse_numeric_frame
from .layout import resolve_benchmark_output_layout


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark cannot be audited safely before training."""


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically for hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "benchmark"


def _require_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise BenchmarkConfigError(f"{field} 必须是非空字符串列表")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise BenchmarkConfigError(f"{field} 不能包含空值")
    if len(set(values)) != len(values):
        raise BenchmarkConfigError(f"{field} 不能包含重复列名")
    return values


def _normalise_feature_sets(
    value: Any,
    *,
    descriptors: Sequence[str],
    smiles_cols: Sequence[str],
) -> tuple[Dict[str, Any], ...]:
    """Return a fully explicit feature-set contract.

    ``descriptors`` predates the strict benchmark feature lifecycle.  It is
    retained as a concise compatibility shorthand for global, precomputed
    descriptors.  New YAML may declare ``feature_sets`` so fold-local feature
    transformers (currently the paper-aligned OHE baseline) cannot accidentally
    be precomputed on all rows.
    """
    if value is None:
        return tuple({
            "name": descriptor,
            "algorithm": descriptor,
            "kind": "precomputed_descriptor",
            "component_cols": list(smiles_cols),
            "mode": "concat",
            "params": {},
        } for descriptor in descriptors)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise BenchmarkConfigError("feature_sets 必须是非空列表")

    normalised = []
    names = set()
    known_columns = set(smiles_cols)
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise BenchmarkConfigError("feature_sets[{0}] 必须是 mapping".format(index))
        name = str(item.get("name", "")).strip().lower()
        if not name:
            raise BenchmarkConfigError("feature_sets[{0}].name 不能为空".format(index))
        if name in names:
            raise BenchmarkConfigError("feature_sets.name 不能重复：{0}".format(name))
        kind = str(item.get("kind", "")).strip()
        if kind not in {"precomputed_descriptor", "fold_transform"}:
            raise BenchmarkConfigError(
                "feature_sets[{0}].kind 必须为 precomputed_descriptor 或 fold_transform".format(index)
            )
        component_cols = _require_string_list(
            item.get("component_cols", smiles_cols),
            "feature_sets[{0}].component_cols".format(index),
        )
        unknown = [column for column in component_cols if column not in known_columns]
        if unknown:
            raise BenchmarkConfigError(
                "feature_sets[{0}].component_cols 不属于 smiles_cols：{1}".format(
                    index, ", ".join(unknown)
                )
            )
        params = item.get("params", {})
        if not isinstance(params, Mapping):
            raise BenchmarkConfigError("feature_sets[{0}].params 必须是 mapping".format(index))
        algorithm = str(item.get("algorithm", name)).strip().lower()
        if not algorithm:
            raise BenchmarkConfigError("feature_sets[{0}].algorithm 不能为空".format(index))
        mode = str(item.get("mode", "concat")).strip()
        if mode not in {"concat", "sum"}:
            raise BenchmarkConfigError("feature_sets[{0}].mode 必须为 concat 或 sum".format(index))
        normalised.append({
            "name": name,
            "algorithm": algorithm,
            "kind": kind,
            "component_cols": list(component_cols),
            "mode": mode,
            "params": json.loads(_canonical_json(dict(params))),
        })
        names.add(name)

    if descriptors and tuple(names) != tuple(descriptors):
        # Supplying both fields is intentionally allowed during migration, but
        # not when their candidate matrices disagree.
        if set(names) != set(descriptors):
            raise BenchmarkConfigError(
                "descriptors 与 feature_sets 的名称集合不一致；请只保留 descriptors，"
                "或让二者表达同一候选特征集"
            )
    return tuple(normalised)


@dataclass(frozen=True)
class BenchmarkConfig:
    """经 schema 校验且路径已解析的 benchmark 配置。"""

    source_path: Path
    dataset_path: Path
    sample_id_col: str
    label_col: str
    smiles_cols: tuple[str, ...]
    descriptors: tuple[str, ...]
    feature_sets: tuple[Dict[str, Any], ...]
    models: tuple[str, ...]
    grouping: Dict[str, Any]
    cv: Dict[str, Any]
    artifact_output_dir: Path
    outputs_root: Path
    split_manifest_path: Path | None
    model_configs: Dict[str, Dict[str, Any]]
    legacy_model_kwargs: Dict[str, Dict[str, Any]]
    task_state: Dict[str, Any]
    raw: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path | str) -> "BenchmarkConfig":
        """Load the only supported executable strict-benchmark YAML.

        The former ``benchmark:`` root is intentionally not adapted here.  It
        had independent model-parameter, grouping and output conventions, so
        accepting it would make a failed migration look like an ordinary
        ``outer_kfold`` run.  Paper-exact archival material remains isolated
        behind :meth:`from_paper_exact_json` and never reaches this launch
        path.
        """
        source_path = Path(path).resolve()
        try:
            loaded = load_run_config(source_path)
        except (ConfigLoadError, ConfigContractError) as exc:
            raise BenchmarkConfigError(str(exc)) from exc
        effective = loaded.effective
        if effective.get("stage") != "benchmark":
            raise BenchmarkConfigError(
                "严格 benchmark 配置必须声明 stage: benchmark；普通 outer_kfold 配置不能替代 manifest 外部折协议"
            )
        return cls._from_schema2(source_path, effective)

    @classmethod
    def from_paper_exact_json(cls, path: Path | str) -> "BenchmarkConfig":
        """Load pre-existing paper-exact archival material only.

        This is not a runnable configuration compatibility entry point.  The
        paper-exact modules call it while reading their already-versioned JSON
        materials; public ``_verify/run_benchmark.py`` and ``yonod.py`` use
        :meth:`from_file` exclusively.
        """
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"paper-exact 配置文件不存在：{source_path}")
        if source_path.suffix.lower() != ".json":
            raise BenchmarkConfigError("paper-exact 归档材料必须是既有 .json 文件；新运行请使用 schema-2 YAML")
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BenchmarkConfigError(f"paper-exact JSON 无法解析：{exc}") from exc
        if not isinstance(raw, Mapping):
            raise BenchmarkConfigError("paper-exact JSON 根节点必须是 mapping")
        block = raw.get("benchmark", raw)
        if not isinstance(block, Mapping):
            raise BenchmarkConfigError("benchmark 节点必须是 mapping")
        protocol = str(block.get("evaluation_protocol", ""))
        reproduction = block.get("reproduction_protocol", {})
        reproduction_name = str(reproduction.get("name", "")) if isinstance(reproduction, Mapping) else ""
        reproduction_protocol = str(reproduction.get("evaluation_protocol", "")) if isinstance(reproduction, Mapping) else ""
        if (
            protocol != "paper_exact_5x5"
            and reproduction_protocol != "paper_exact_5x5"
            and reproduction_name != "vjethbkm_rf_5x5"
        ):
            raise BenchmarkConfigError(
                "仅 paper-exact 归档材料可使用 from_paper_exact_json；普通 benchmark 必须迁移为 schema-2 YAML"
            )
        return cls._from_legacy_paper_exact_block(source_path, block)

    @classmethod
    def _from_legacy_paper_exact_block(cls, source_path: Path, block: Mapping[str, Any]) -> "BenchmarkConfig":

        required = ("dataset_path", "sample_id_col", "label_col", "smiles_cols", "models")
        missing = [key for key in required if not block.get(key)]
        if missing:
            raise BenchmarkConfigError(f"配置缺少必填字段：{', '.join(missing)}")
        if not block.get("descriptors") and not block.get("feature_sets"):
            raise BenchmarkConfigError("配置至少需要 descriptors 或 feature_sets")

        dataset_path = Path(str(block["dataset_path"]))
        if not dataset_path.is_absolute():
            dataset_path = (source_path.parent / dataset_path).resolve()
        outputs = block.get("outputs", {})
        if not isinstance(outputs, Mapping):
            raise BenchmarkConfigError("outputs 必须是 mapping")
        outputs_root = Path(str(outputs.get("root", "results")))
        if not outputs_root.is_absolute():
            outputs_root = (source_path.parent / outputs_root).resolve()

        grouping = dict(block.get("grouping", {}))
        cv = dict(block.get("cv", {}))
        if not grouping.get("strategy"):
            raise BenchmarkConfigError("grouping.strategy 不能为空")
        for field in ("n_repeats", "n_splits", "seed"):
            if field not in cv:
                raise BenchmarkConfigError(f"cv.{field} 不能为空")
        if int(cv["n_repeats"]) < 1 or int(cv["n_splits"]) < 2:
            raise BenchmarkConfigError("cv.n_repeats 必须 >= 1，cv.n_splits 必须 >= 2")

        smiles_cols = _require_string_list(block["smiles_cols"], "smiles_cols")
        descriptors = (
            _require_string_list(block["descriptors"], "descriptors")
            if block.get("descriptors")
            else tuple()
        )
        feature_sets = _normalise_feature_sets(
            block.get("feature_sets"),
            descriptors=descriptors,
            smiles_cols=smiles_cols,
        )
        # Existing callers use ``descriptors`` as a feature-set identity.  Keep
        # that public compatibility surface while allowing fold-local sets such
        # as OHE to join the same benchmark matrix.
        active_feature_names = tuple(str(item["name"]) for item in feature_sets)

        return cls(
            source_path=source_path,
            dataset_path=dataset_path,
            sample_id_col=str(block["sample_id_col"]),
            label_col=str(block["label_col"]),
            smiles_cols=smiles_cols,
            descriptors=active_feature_names,
            feature_sets=feature_sets,
            models=_require_string_list(block["models"], "models"),
            grouping=grouping,
            cv=cv,
            artifact_output_dir=outputs_root,
            outputs_root=outputs_root,
            split_manifest_path=None,
            model_configs={},
            legacy_model_kwargs={
                str(name): dict(params)
                for name, params in dict(block.get("model_params", {})).items()
                if isinstance(params, Mapping)
            },
            task_state={"backend": "sqlite", "resumable": True, "legacy_paper_exact": True},
            raw=dict(block),
        )

    @classmethod
    def _from_schema2(cls, source_path: Path, raw: Mapping[str, Any]) -> "BenchmarkConfig":
        """Adapt a validated schema-2 declaration without changing protocol."""
        dataset = raw["dataset"]
        roles = dataset["column_roles"]
        label_col = str(roles["label"])
        molecular_columns: list[str] = []
        for role in ("reactants", "products", "others"):
            molecular_columns.extend(str(column) for column in roles.get(role, []) or [])
        smiles_cols = _require_string_list(
            molecular_columns,
            "dataset.column_roles.reactants/products/others",
        )
        dataset_path = resolve_config_path(source_path, str(dataset["path"]))
        artifact_output_dir = resolve_config_path(source_path, str(raw["artifacts"]["output_dir"]))
        outputs = raw.get("outputs")
        if not isinstance(outputs, Mapping) or not isinstance(outputs.get("root"), str) or not outputs["root"].strip():
            raise BenchmarkConfigError("stage: benchmark 必须声明非空 outputs.root")
        outputs_root = resolve_config_path(source_path, str(outputs["root"]))
        try:
            artifact_output_dir.relative_to(outputs_root)
        except ValueError as exc:
            raise BenchmarkConfigError(
                "严格 benchmark 的 artifacts.output_dir 必须位于 outputs.root 任务目录内；"
                "不能把特征写到其他任务或 config/ 下"
            ) from exc
        if artifact_output_dir == outputs_root:
            raise BenchmarkConfigError(
                "严格 benchmark 的 artifacts.output_dir 必须是 outputs.root 的子目录（标准为 ./result/<task>/feature）"
            )
        # The current strict layout stores static descriptor packages in the
        # same documented ``feature`` child as the schema-2 feature service.
        # Reject another child rather than silently ignoring a valid-looking
        # user path.
        if artifact_output_dir != outputs_root / "feature":
            raise BenchmarkConfigError(
                "严格 benchmark 当前要求 artifacts.output_dir 解析为 outputs.root/feature；"
                "该目录既是可复用特征包位置，也是任务身份的一部分"
            )

        evaluation = raw.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise BenchmarkConfigError("stage: benchmark 必须声明 evaluation")
        if evaluation.get("protocol") != "manifest_outer_cv":
            raise BenchmarkConfigError(
                "stage: benchmark 只支持 evaluation.protocol: manifest_outer_cv；不能降级为 outer_kfold"
            )
        split_manifest_path = None
        if evaluation.get("split_manifest") is not None:
            split_manifest_path = resolve_config_path(source_path, str(evaluation["split_manifest"]))
            if not split_manifest_path.is_file():
                raise BenchmarkConfigError(
                    f"evaluation.split_manifest 不存在：{split_manifest_path}"
                )
        grouping_value = evaluation.get("grouping")
        if not isinstance(grouping_value, Mapping) or not grouping_value.get("strategy"):
            raise BenchmarkConfigError("evaluation.grouping.strategy 不能为空；严格 benchmark 不会回退为普通 outer_kfold")
        grouping = copy.deepcopy(dict(grouping_value))
        strategy = str(grouping.get("strategy", "")).strip()
        supported_grouping = {
            "repeated_kfold", "component_holdout", "substrate_scaffold",
            "reaction_fingerprint_cluster", "precomputed_column",
        }
        if strategy not in supported_grouping:
            raise BenchmarkConfigError(
                "evaluation.grouping.strategy 不受支持：{0}；可用值：{1}".format(
                    strategy, ", ".join(sorted(supported_grouping))
                )
            )
        if strategy == "precomputed_column":
            group_column = grouping.get("group_column")
            if not isinstance(group_column, str) or not group_column.strip():
                raise BenchmarkConfigError(
                    "evaluation.grouping.strategy=precomputed_column 需要非空 group_column"
                )
        cv: Dict[str, Any] = {}
        for field in ("n_repeats", "n_splits", "seed"):
            if field not in evaluation:
                raise BenchmarkConfigError(f"evaluation.{field} 不能为空")
            value = evaluation[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise BenchmarkConfigError(f"evaluation.{field} 必须是整数")
            cv[field] = int(value)
        if cv["n_repeats"] < 1 or cv["n_splits"] < 2:
            raise BenchmarkConfigError("evaluation.n_repeats 必须 >= 1，evaluation.n_splits 必须 >= 2")

        feature_sets = cls._schema2_feature_sets(raw["descriptors"], smiles_cols)
        active_feature_names = tuple(str(item["name"]) for item in feature_sets)
        model_configs, legacy_model_kwargs = cls._schema2_model_configs(raw.get("model_params", {}), raw["models"])
        numeric_contract = normalise_numeric_contract(dataset)
        if numeric_contract["columns"] and "autogluon" in raw["models"]:
            raise BenchmarkConfigError(
                "strict AutoGluon 的内部验证/集成子折尚不能保证数值处理只拟合子折训练行"
            )
        benchmark = raw["benchmark"]
        task_state = copy.deepcopy(dict(benchmark["task_state"]))
        internal_raw: Dict[str, Any] = {
            "schema_version": "2.0",
            "hpo": copy.deepcopy(raw.get("hpo", {"enabled": False})),
            "model_params": legacy_model_kwargs,
            "schema2_model_params": copy.deepcopy(model_configs),
            "reproduction_protocol": copy.deepcopy(dict(benchmark.get("reproduction_protocol", {}))),
            "evaluation_protocol": "manifest_outer_cv",
            "task_state": task_state,
            "project_name": raw["project_name"],
            "numeric_contract": numeric_contract,
        }
        if raw.get("hpo", {}).get("enabled"):
            internal_raw["dataset_roles"] = copy.deepcopy(dict(roles))
        for field in ("population_id", "dataset_id"):
            if field in benchmark:
                internal_raw[field] = benchmark[field]
        return cls(
            source_path=source_path,
            dataset_path=dataset_path,
            sample_id_col=str(dataset["sample_id_col"]),
            label_col=label_col,
            smiles_cols=smiles_cols,
            descriptors=active_feature_names,
            feature_sets=feature_sets,
            models=_require_string_list(raw["models"], "models"),
            grouping=grouping,
            cv=cv,
            artifact_output_dir=artifact_output_dir,
            outputs_root=outputs_root,
            split_manifest_path=split_manifest_path,
            model_configs=model_configs,
            legacy_model_kwargs=legacy_model_kwargs,
            task_state=task_state,
            raw=internal_raw,
        )

    @staticmethod
    def _schema2_feature_sets(value: Any, smiles_cols: Sequence[str]) -> tuple[Dict[str, Any], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise BenchmarkConfigError("stage: benchmark 的 descriptors 必须是非空列表")
        feature_sets = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, Mapping):
                raise BenchmarkConfigError("descriptors[{0}] 必须是 mapping".format(index))
            # Feature IDs are part of the task and artifact identity.  Do not
            # case-fold them while adapting schema-2: doing so could silently
            # merge two distinct user-declared feature sets.
            feature_id = str(item.get("id", "")).strip()
            algorithm = str(item.get("descriptor", "")).strip().lower()
            lifecycle = str(item.get("lifecycle", "")).strip()
            if not feature_id or not algorithm:
                raise BenchmarkConfigError("descriptors[{0}] 必须声明 id 和 descriptor".format(index))
            if lifecycle not in {"static_descriptor", "fold_transform"}:
                raise BenchmarkConfigError(
                    "descriptors[{0}].lifecycle 必须显式为 static_descriptor 或 fold_transform".format(index)
                )
            columns = _require_string_list(item.get("columns"), "descriptors[{0}].columns".format(index))
            unknown = [column for column in columns if column not in smiles_cols]
            if unknown:
                raise BenchmarkConfigError(
                    "descriptors[{0}].columns 必须属于 dataset.column_roles 的 reactants/products/others：{1}".format(
                        index, ", ".join(unknown)
                    )
                )
            if item.get("extra_reactants"):
                raise BenchmarkConfigError(
                    "严格 benchmark 尚不能逐字段映射 descriptors[{0}].extra_reactants；"
                    "为避免改变组件顺序而拒绝运行".format(index)
                )
            if lifecycle == "fold_transform" and algorithm != "ohe":
                raise BenchmarkConfigError(
                    "严格 benchmark 目前唯一支持的 fold_transform descriptor 是 ohe；收到 {0!r}".format(algorithm)
                )
            params = item.get("params", {})
            if not isinstance(params, Mapping):
                raise BenchmarkConfigError("descriptors[{0}].params 必须是 mapping".format(index))
            mode = str(item.get("mode", "concat")).strip()
            if mode not in {"concat", "sum"}:
                raise BenchmarkConfigError("descriptors[{0}].mode 必须为 concat 或 sum".format(index))
            feature_sets.append({
                "name": feature_id,
                "algorithm": algorithm,
                "kind": "fold_transform" if lifecycle == "fold_transform" else "precomputed_descriptor",
                "component_cols": list(columns),
                "mode": mode,
                "params": copy.deepcopy(dict(params)),
            })
        names = [str(item["name"]) for item in feature_sets]
        if len(set(names)) != len(names):
            raise BenchmarkConfigError("descriptors 的 id 不能重复")
        return tuple(feature_sets)

    @staticmethod
    def _schema2_model_configs(value: Any, models: Any) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if not isinstance(value, Mapping):
            raise BenchmarkConfigError("model_params 必须是 mapping")
        model_names = _require_string_list(models, "models")
        unknown = sorted(str(name) for name in value if str(name) not in model_names)
        if unknown:
            raise BenchmarkConfigError("model_params 包含未列入 models 的模型：{0}".format(", ".join(unknown)))
        configs: Dict[str, Dict[str, Any]] = {}
        legacy_kwargs: Dict[str, Dict[str, Any]] = {}
        for model in model_names:
            section = value.get(model, {})
            if not isinstance(section, Mapping):
                raise BenchmarkConfigError("model_params.{0} 必须是 mapping".format(model))
            try:
                resolved = resolve_model_config(model, section)
            except ModelConfigurationError as exc:
                raise BenchmarkConfigError(str(exc)) from exc
            if resolved.fit and model != "autogluon":
                raise BenchmarkConfigError(
                    "严格 manifest 外部折尚未实现 model_params.{0}.fit 的逐折数据列物化；为避免改变样本权重或验证边界而拒绝运行".format(model)
                )
            if model in {"xgb", "lightgbm"} and "early_stopping" in resolved.runtime:
                raise BenchmarkConfigError(
                    "严格 benchmark 尚未为 runtime.early_stopping 建立已审计的内层验证折；请勿把它降级为外层 eval_set"
                )
            if model == "autogluon":
                unsupported_predictor = set(resolved.estimator).difference({"label"})
                unsupported_fit = set(resolved.fit).difference({"time_limit", "presets", "num_cpus"})
                if unsupported_predictor or unsupported_fit:
                    raise BenchmarkConfigError(
                        "严格 benchmark 的 AutoGluon adapter 仅支持 fit.time_limit、fit.presets、fit.num_cpus；"
                        "不支持的 predictor/fit 字段：{0}".format(
                            ", ".join(sorted(unsupported_predictor | unsupported_fit))
                        )
                    )
                if resolved.estimator.get("label") not in {None, "yield"}:
                    raise BenchmarkConfigError(
                        "严格 benchmark AutoGluon adapter 的内部标签字段固定为 yield；请不要声明 predictor.label"
                    )
                legacy_kwargs[model] = {
                    **dict(resolved.fit),
                    **dict(resolved.runtime),
                }
            configs[model] = copy.deepcopy(dict(section))
        return configs, legacy_kwargs

    def validate_dataset(self) -> pd.DataFrame:
        """Read and validate the data contract before any training begins."""
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"数据集不存在：{self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        numeric_contract = self.raw.get("numeric_contract", {"columns": []})
        required = [self.sample_id_col, self.label_col, *self.smiles_cols,
                    *[entry["source"] for entry in numeric_contract["columns"]]]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise BenchmarkConfigError(f"数据集缺少列：{', '.join(missing)}")
        sample_ids = frame[self.sample_id_col]
        if sample_ids.isna().any() or sample_ids.astype(str).str.strip().eq("").any():
            raise BenchmarkConfigError(f"sample_id 列 {self.sample_id_col!r} 存在空值")
        if sample_ids.duplicated().any():
            duplicates = sample_ids[sample_ids.duplicated()].head(5).tolist()
            raise BenchmarkConfigError(
                f"sample_id 列 {self.sample_id_col!r} 不唯一，示例重复值：{duplicates}"
            )
        labels = pd.to_numeric(frame[self.label_col], errors="coerce")
        if labels.isna().any():
            bad_count = int(labels.isna().sum())
            raise BenchmarkConfigError(f"标签列 {self.label_col!r} 含 {bad_count} 个不可解析或空值")
        if numeric_contract["columns"]:
            source_columns = [entry["source"] for entry in numeric_contract["columns"]]
            raw_numeric = pd.read_csv(
                self.dataset_path, usecols=[self.sample_id_col, *source_columns], keep_default_na=False
            )
            if len(raw_numeric) != len(frame) or raw_numeric[self.sample_id_col].astype(str).tolist() != sample_ids.astype(str).tolist():
                raise BenchmarkConfigError("数值原始列与 strict sample_id/行序不一致")
            try:
                parse_numeric_frame(raw_numeric, sample_ids.astype(str).tolist(), numeric_contract, phase="strict_preflight")
            except NumericConditionsError as exc:
                raise BenchmarkConfigError(str(exc)) from exc
            for column in source_columns:
                frame[column] = raw_numeric[column].tolist()
        return frame

    def normalized_for_hash(self, dataset_sha256: str) -> Dict[str, Any]:
        """Return every result-affecting setting in a deterministic structure."""
        normalized = {
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": dataset_sha256,
            "sample_id_col": self.sample_id_col,
            "label_col": self.label_col,
            "smiles_cols": list(self.smiles_cols),
            "descriptors": list(self.descriptors),
            "feature_sets": [dict(item) for item in self.feature_sets],
            "models": list(self.models),
            "model_params": self.raw.get("model_params", {}),
            "schema2_model_params": self.raw.get("schema2_model_params", {}),
            "reproduction_protocol": self.raw.get("reproduction_protocol", {}),
            "grouping": self.grouping,
            "cv": self.cv,
            "artifact_output_dir": str(self.artifact_output_dir),
            "outputs_root": str(self.outputs_root),
            "task_state": self.task_state,
            "numeric_contract": self.raw.get("numeric_contract"),
            "external_split_manifest_sha256": (
                _sha256_file(self.split_manifest_path)
                if self.split_manifest_path is not None else None
            ),
        }
        if self.raw.get("hpo", {}).get("enabled"):
            normalized["hpo"] = self.raw["hpo"]
        for optional_field in ("population_id", "dataset_id", "paper_exact", "evaluation_protocol"):
            if optional_field in self.raw:
                normalized[optional_field] = self.raw[optional_field]
        return normalized


@dataclass(frozen=True)
class BenchmarkContract:
    """A run-ready, reproducible identity for one benchmark execution."""

    config: BenchmarkConfig
    dataset_sha256: str
    config_hash: str
    run_id: str
    run_dir: Path
    manifest: Dict[str, Any]


def create_benchmark_contract(config: BenchmarkConfig) -> BenchmarkContract:
    """Validate data and build a deterministic run identity without writing files."""
    config.validate_dataset()
    dataset_sha256 = _sha256_file(config.dataset_path)
    normalized = config.normalized_for_hash(dataset_sha256)
    config_hash = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    run_id = f"{_slug(config.dataset_path.stem)}-{config_hash[:12]}"
    project_root = config.source_path.parent
    for parent in config.source_path.parents:
        if (parent / ".git").exists():
            project_root = parent
            break
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset_sha256": dataset_sha256,
        "dataset_path": str(config.dataset_path),
        "benchmark_config": normalized,
        "source_config_path": str(config.source_path),
        "code_git_commit": _git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_layout": {
            "descriptors": "feature",
            "docs": "docs",
            "pictures": "pictures",
            "report": "report",
        },
    }
    if config.split_manifest_path is not None:
        manifest["external_split_manifest"] = {
            "source_path": str(config.split_manifest_path),
            "sha256": _sha256_file(config.split_manifest_path),
            "import_policy": "validated_membership_then_task_local_run_id",
        }
    for optional_field in ("population_id", "dataset_id", "paper_exact", "evaluation_protocol"):
        if optional_field in config.raw:
            manifest[optional_field] = config.raw[optional_field]
    return BenchmarkContract(
        config=config,
        dataset_sha256=dataset_sha256,
        config_hash=config_hash,
        run_id=run_id,
        # ``outputs.root`` is already the unique task directory required by
        # schema-2.  Adding a content-derived child made a legal
        # ``artifacts.output_dir: .../feature`` impossible to honour and
        # scattered one task across two roots.  The immutable run/config hash
        # in the manifest still rejects incompatible reuse of this directory.
        run_dir=config.outputs_root,
        manifest=manifest,
    )


def write_run_manifest(contract: BenchmarkContract) -> Path:
    """Persist the run manifest without allowing an incompatible overwrite.

    A repeated invocation for the same config is safe and supports later resume;
    a conflicting file at the deterministic run path is treated as corruption.
    """
    manifest_path = resolve_benchmark_output_layout(contract.run_dir).manifests / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("config_hash") != contract.config_hash:
            raise BenchmarkConfigError(
                f"运行目录已被不同配置占用：{contract.run_dir}；请使用新的 outputs.root"
            )
        return manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(contract.manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(manifest_path)
    return manifest_path

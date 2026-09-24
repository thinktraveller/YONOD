"""Audited frozen-training / external-test execution.

This module is intentionally separate from ``manifest_outer_cv``.  It fits
once on the declared development population and predicts a different,
checksum-pinned population.  There are no folds, split manifests, CV summary,
or ranking semantics to accidentally reuse.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config.contracts import ConfigContractError, resolve_config_path
from ..config.loader import ConfigLoadError, load_run_config
from ..model_factory import ModelConfigurationError, ResolvedModelConfig, construct_estimator, resolve_model_config
from ..universal.feature_builder import build_universal_features


EXTERNAL_PROTOCOL = "frozen_train_external_test"


class FrozenExternalTestError(ValueError):
    """Raised before any model fit when an external-test contract is unsafe."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_values(values: Sequence[Any]) -> str:
    return hashlib.sha256(
        _canonical_json([str(value) for value in values]).encode("utf-8")
    ).hexdigest()


def _git_commit(project_root: Path) -> str | None:
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


def _atomic_json(payload: Mapping[str, Any], path: Path) -> Path:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise FrozenExternalTestError("输出已经存在且内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_text(text: str, path: Path) -> Path:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise FrozenExternalTestError("输出已经存在且内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> Path:
    if path.exists():
        existing = pd.read_parquet(path)
        if existing.equals(frame):
            return path
        raise FrozenExternalTestError("输出已经存在且内容不同，拒绝覆盖：{0}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    verified = pd.read_parquet(temporary)
    if list(verified.columns) != list(frame.columns) or len(verified) != len(frame):
        temporary.unlink(missing_ok=True)
        raise FrozenExternalTestError("外部预测临时 parquet 校验失败")
    temporary.replace(path)
    return path


def _require_unique_text(frame: pd.DataFrame, column: str, context: str) -> pd.Series:
    values = frame[column]
    normalized = values.astype(str).str.strip()
    if values.isna().any() or normalized.eq("").any():
        raise FrozenExternalTestError("{0} 的 {1!r} 含空值".format(context, column))
    if normalized.duplicated().any():
        examples = normalized[normalized.duplicated()].head(3).tolist()
        raise FrozenExternalTestError("{0} 的 {1!r} 不唯一，示例：{2}".format(context, column, examples))
    return normalized


def _read_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenExternalTestError("无法读取 {0}：{1}".format(context, exc)) from exc
    if not isinstance(payload, dict):
        raise FrozenExternalTestError("{0} 的 JSON 根节点必须是对象".format(context))
    return payload


def _normalise_report_formats(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FrozenExternalTestError("outputs.report_formats 必须同时声明 html 和 markdown")
    normalized = tuple("markdown" if str(item).lower() == "md" else str(item).lower() for item in value)
    if set(normalized) != {"html", "markdown"} or len(normalized) != 2:
        raise FrozenExternalTestError("frozen_train_external_test 的 outputs.report_formats 必须恰为 [html, markdown]")
    return normalized


@dataclass(frozen=True)
class ExternalFeatureSet:
    name: str
    algorithm: str
    columns: tuple[str, ...]
    mode: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class FrozenExternalTestConfig:
    source_path: Path
    dataset_path: Path
    external_path: Path
    audit_manifest_path: Path
    sample_id_col: str
    label_col: str
    smiles_cols: tuple[str, ...]
    external_sample_id_col: str
    external_label_col: str
    external_source_id_col: str
    external_group_col: str
    external_column_map: Dict[str, str]
    external_expected_sha256: str
    audit_manifest_sha256: str
    feature_sets: tuple[ExternalFeatureSet, ...]
    models: tuple[str, ...]
    model_configs: Dict[str, ResolvedModelConfig]
    seed: int
    artifact_output_dir: Path
    outputs_root: Path
    report_formats: tuple[str, ...]
    raw: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path | str) -> "FrozenExternalTestConfig":
        source_path = Path(path).resolve()
        try:
            effective = load_run_config(source_path).effective
        except (ConfigLoadError, ConfigContractError) as exc:
            raise FrozenExternalTestError(str(exc)) from exc
        if effective.get("stage") != "benchmark":
            raise FrozenExternalTestError("冻结外部测试配置必须声明 stage: benchmark")
        evaluation = effective.get("evaluation")
        if not isinstance(evaluation, Mapping) or evaluation.get("protocol") != EXTERNAL_PROTOCOL:
            raise FrozenExternalTestError(
                "冻结外部测试必须声明 evaluation.protocol: frozen_train_external_test"
            )
        dataset = effective["dataset"]
        roles = dataset["column_roles"]
        molecular_columns: list[str] = []
        for role in ("reactants", "products", "others"):
            molecular_columns.extend(str(column) for column in roles.get(role, []) or [])
        if not molecular_columns or len(set(molecular_columns)) != len(molecular_columns):
            raise FrozenExternalTestError("dataset.column_roles 必须给出不重复的分子特征列")
        artifacts = effective["artifacts"]
        outputs = effective.get("outputs")
        if not isinstance(outputs, Mapping) or not isinstance(outputs.get("root"), str):
            raise FrozenExternalTestError("冻结外部测试必须声明 outputs.root")
        output_root = resolve_config_path(source_path, str(outputs["root"]))
        artifact_dir = resolve_config_path(source_path, str(artifacts["output_dir"]))
        if output_root.parent.name != "result" or artifact_dir != output_root / "feature":
            raise FrozenExternalTestError(
                "冻结外部测试必须使用隔离的 result/<task_name>/ 根目录和其 feature/ 子目录"
            )
        external = evaluation.get("external_test")
        if not isinstance(external, Mapping):  # generic schema normally catches this
            raise FrozenExternalTestError("evaluation.external_test 必须是 mapping")
        column_map = {str(key): str(value) for key, value in dict(external["column_map"]).items()}
        if len(set(column_map.values())) != len(column_map):
            raise FrozenExternalTestError("evaluation.external_test.column_map 必须一对一，不能让两个训练特征复用同一外部列")
        unknown_targets = sorted(set(column_map).difference(molecular_columns))
        if unknown_targets:
            raise FrozenExternalTestError("external_test.column_map 含未知训练特征列：" + ", ".join(unknown_targets))
        feature_sets = cls._feature_sets(effective.get("descriptors"), molecular_columns)
        required_features = {column for item in feature_sets for column in item.columns}
        missing_map = sorted(required_features.difference(column_map))
        if missing_map:
            raise FrozenExternalTestError("external_test.column_map 未覆盖描述符所需列：" + ", ".join(missing_map))
        model_configs = cls._model_configs(effective.get("models"), effective.get("model_params", {}))
        config = cls(
            source_path=source_path,
            dataset_path=resolve_config_path(source_path, str(dataset["path"])),
            external_path=resolve_config_path(source_path, str(external["path"])),
            audit_manifest_path=resolve_config_path(source_path, str(external["audit_manifest"])),
            sample_id_col=str(dataset["sample_id_col"]),
            label_col=str(roles["label"]),
            smiles_cols=tuple(molecular_columns),
            external_sample_id_col=str(external["sample_id_col"]),
            external_label_col=str(external["label_col"]),
            external_source_id_col=str(external["source_id_col"]),
            external_group_col=str(external["group_col"]),
            external_column_map=column_map,
            external_expected_sha256=str(external["expected_sha256"]).lower(),
            audit_manifest_sha256=str(external["audit_manifest_sha256"]).lower(),
            feature_sets=feature_sets,
            models=tuple(str(model) for model in effective["models"]),
            model_configs=model_configs,
            seed=int(evaluation["seed"]),
            artifact_output_dir=artifact_dir,
            outputs_root=output_root,
            report_formats=_normalise_report_formats(outputs.get("report_formats")),
            raw=copy.deepcopy(dict(effective)),
        )
        # This is deliberately part of configuration loading: identity, audit,
        # mapping and train/test disjointness fail before descriptor/model work.
        config.validate_populations()
        return config

    @staticmethod
    def _feature_sets(value: Any, molecular_columns: Sequence[str]) -> tuple[ExternalFeatureSet, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise FrozenExternalTestError("descriptors 必须是非空列表")
        names: set[str] = set()
        items: list[ExternalFeatureSet] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, Mapping):
                raise FrozenExternalTestError("descriptors[{0}] 必须是 mapping".format(index))
            name = str(item.get("id", "")).strip()
            algorithm = str(item.get("descriptor", "")).strip().lower()
            lifecycle = str(item.get("lifecycle", "")).strip()
            columns = tuple(str(column) for column in item.get("columns", []) or [])
            if not name or not algorithm or not columns:
                raise FrozenExternalTestError("descriptors[{0}] 必须声明 id、descriptor 和 columns".format(index))
            if name in names:
                raise FrozenExternalTestError("descriptors.id 不能重复：{0}".format(name))
            if lifecycle != "static_descriptor":
                raise FrozenExternalTestError("冻结外部测试仅支持 static_descriptor；fold_transform 没有合法的外部拟合边界")
            if item.get("extra_reactants"):
                raise FrozenExternalTestError("冻结外部测试不支持 descriptors.extra_reactants")
            unknown = sorted(set(columns).difference(molecular_columns))
            if unknown:
                raise FrozenExternalTestError("descriptors[{0}].columns 不属于 dataset roles：{1}".format(index, ", ".join(unknown)))
            mode = str(item.get("mode", "concat"))
            if mode not in {"concat", "sum"}:
                raise FrozenExternalTestError("descriptors[{0}].mode 必须是 concat 或 sum".format(index))
            params = item.get("params", {})
            if not isinstance(params, Mapping):
                raise FrozenExternalTestError("descriptors[{0}].params 必须是 mapping".format(index))
            items.append(ExternalFeatureSet(name, algorithm, columns, mode, copy.deepcopy(dict(params))))
            names.add(name)
        return tuple(items)

    @staticmethod
    def _model_configs(models: Any, params: Any) -> Dict[str, ResolvedModelConfig]:
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)) or not models:
            raise FrozenExternalTestError("models 必须是非空列表")
        if not isinstance(params, Mapping):
            raise FrozenExternalTestError("model_params 必须是 mapping")
        resolved: Dict[str, ResolvedModelConfig] = {}
        for model_value in models:
            model = str(model_value)
            if model == "autogluon":
                raise FrozenExternalTestError("冻结外部测试尚未实现 AutoGluon 的无外部标签适配器")
            try:
                model_config = resolve_model_config(model, params.get(model, {}))
            except ModelConfigurationError as exc:
                raise FrozenExternalTestError("模型 {0} 配置无效：{1}".format(model, exc)) from exc
            if model_config.fit:
                raise FrozenExternalTestError("冻结外部测试禁止 model_params.{0}.fit；训练只接受 X_train/y_train".format(model))
            if "early_stopping" in model_config.runtime:
                raise FrozenExternalTestError("冻结外部测试不支持外部测试参与 early stopping；请先建立独立的训练内层协议")
            if model in {"rf", "xgb", "lightgbm"} and model_config.estimator.get("n_jobs") != 19:
                raise FrozenExternalTestError("model_params.{0}.estimator.n_jobs 必须为 19".format(model))
            resolved[model] = model_config
        return resolved

    def validate_populations(self) -> Dict[str, Any]:
        if not self.dataset_path.is_file() or not self.external_path.is_file() or not self.audit_manifest_path.is_file():
            missing = [str(path) for path in (self.dataset_path, self.external_path, self.audit_manifest_path) if not path.is_file()]
            raise FrozenExternalTestError("冻结外部测试所需文件不存在：" + ", ".join(missing))
        if _sha256_file(self.external_path) != self.external_expected_sha256:
            raise FrozenExternalTestError("外部测试数据 SHA-256 与 YAML 固定值不一致")
        if _sha256_file(self.audit_manifest_path) != self.audit_manifest_sha256:
            raise FrozenExternalTestError("外部审计清单 SHA-256 与 YAML 固定值不一致")
        audit = _read_json(self.audit_manifest_path, "外部审计清单")
        source_sha = audit.get("candidate_source_sha256")
        if not isinstance(source_sha, str) or len(source_sha) != 64:
            raise FrozenExternalTestError("外部审计清单缺少 candidate_source_sha256")
        overlap = audit.get("exact_canonical_overlap")
        if not isinstance(overlap, Mapping) or any(int(overlap.get(name, -1)) != 0 for name in ("acid", "amine", "product")):
            raise FrozenExternalTestError("外部审计清单未证明 A/B/P canonical overlap 均为零")

        train = pd.read_csv(self.dataset_path)
        required_train = [self.sample_id_col, self.label_col, *self.smiles_cols]
        missing_train = sorted(set(required_train).difference(train.columns))
        if missing_train:
            raise FrozenExternalTestError("训练数据缺少列：" + ", ".join(missing_train))
        train_ids = _require_unique_text(train, self.sample_id_col, "训练数据")
        train_y = pd.to_numeric(train[self.label_col], errors="coerce")
        if train_y.isna().any() or not np.isfinite(train_y.to_numpy(dtype=float)).all():
            raise FrozenExternalTestError("训练标签含不可解析或非有限值")

        header = pd.read_csv(self.external_path, nrows=0)
        required_external = [
            self.external_sample_id_col, self.external_label_col, self.external_source_id_col,
            self.external_group_col, *self.external_column_map.values(),
        ]
        missing_external = sorted(set(required_external).difference(header.columns))
        if missing_external:
            raise FrozenExternalTestError("外部测试数据缺少列：" + ", ".join(missing_external))
        # Do not read external labels in this pre-fit phase.  Only IDs, source
        # IDs, groups and declared feature columns are needed for leakage tests.
        identity_columns = list(dict.fromkeys([
            self.external_sample_id_col, self.external_source_id_col, self.external_group_col,
            *self.external_column_map.values(),
        ]))
        external = pd.read_csv(self.external_path, usecols=identity_columns)
        external_ids = _require_unique_text(external, self.external_sample_id_col, "外部测试数据")
        source_ids = _require_unique_text(external, self.external_source_id_col, "外部测试数据")
        group_ids = external[self.external_group_col].astype(str).str.strip()
        if external[self.external_group_col].isna().any() or group_ids.eq("").any() or group_ids.nunique() < 2:
            raise FrozenExternalTestError("外部测试 group_col 必须含至少两个非空组，供后续 cluster bootstrap 使用")
        overlap_ids = sorted(set(train_ids).intersection(external_ids))
        if overlap_ids:
            raise FrozenExternalTestError("训练/外部 sample_id 发生泄漏重叠，示例：" + ", ".join(overlap_ids[:3]))
        expected_rows = audit.get("row_count")
        if isinstance(expected_rows, int) and expected_rows != len(external):
            raise FrozenExternalTestError("外部行数与审计清单 row_count 不一致")
        physical_dataset_id = audit.get("physical_dataset_id")
        if isinstance(physical_dataset_id, str) and physical_dataset_id:
            prefix = physical_dataset_id + ":"
            if not source_ids.str.startswith(prefix).all():
                raise FrozenExternalTestError("外部 source_id 未全部绑定到审计清单的 physical_dataset_id")
        return {
            "audit": audit,
            "n_train": int(len(train)),
            "n_external": int(len(external)),
            "n_external_groups": int(group_ids.nunique()),
            "train_sample_id_sha256": _hash_values(train_ids.tolist()),
            "external_sample_id_sha256": _hash_values(external_ids.tolist()),
            "external_source_id_sha256": _hash_values(source_ids.tolist()),
        }

    def normalized_for_hash(self, population: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "evaluation_protocol": EXTERNAL_PROTOCOL,
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": _sha256_file(self.dataset_path),
            "external_path": str(self.external_path),
            "external_sha256": self.external_expected_sha256,
            "audit_manifest_path": str(self.audit_manifest_path),
            "audit_manifest_sha256": self.audit_manifest_sha256,
            "sample_id_col": self.sample_id_col,
            "label_col": self.label_col,
            "smiles_cols": list(self.smiles_cols),
            "external_test": {
                "sample_id_col": self.external_sample_id_col,
                "label_col": self.external_label_col,
                "source_id_col": self.external_source_id_col,
                "group_col": self.external_group_col,
                "column_map": dict(self.external_column_map),
            },
            "feature_sets": [
                {"name": item.name, "algorithm": item.algorithm, "columns": list(item.columns), "mode": item.mode, "params": item.params}
                for item in self.feature_sets
            ],
            "models": list(self.models),
            "model_params": self.raw.get("model_params", {}),
            "seed": self.seed,
            "artifact_output_dir": str(self.artifact_output_dir),
            "outputs_root": str(self.outputs_root),
            "population_identity": dict(population),
        }


@dataclass(frozen=True)
class FrozenExternalContract:
    config: FrozenExternalTestConfig
    config_hash: str
    run_id: str
    manifest: Dict[str, Any]
    population: Dict[str, Any]


def create_frozen_external_contract(config: FrozenExternalTestConfig) -> FrozenExternalContract:
    population = config.validate_populations()
    normalized = config.normalized_for_hash(population)
    config_hash = hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()
    project_root = config.source_path.parent
    for candidate in config.source_path.parents:
        if (candidate / ".git").exists():
            project_root = candidate
            break
    run_id = "frozen-external-" + config_hash[:12]
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "config_hash": config_hash,
        "evaluation_protocol": EXTERNAL_PROTOCOL,
        "single_fit": True,
        "cv_folds": 0,
        "strict_rank_eligible": False,
        "strict_rank_exclusion_reason": "single frozen-development/external-test evaluation is not CV",
        "benchmark_config": normalized,
        "source_config_path": str(config.source_path),
        "code_git_commit": _git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_isolation": "External labels are loaded only after model predictions are written in memory.",
        "output_layout": {"docs": "docs", "predictions": "docs/predictions", "metrics": "docs/metrics", "report": "report"},
    }
    return FrozenExternalContract(config, config_hash, run_id, manifest, population)


def _layout(root: Path) -> Dict[str, Path]:
    return {
        "root": root,
        "feature": root / "feature",
        "docs": root / "docs",
        "manifests": root / "docs" / "manifests",
        "predictions": root / "docs" / "predictions",
        "folds": root / "docs" / "folds",
        "metrics": root / "docs" / "metrics",
        "report": root / "report",
    }


def _canonical_external_feature_frame(config: FrozenExternalTestConfig) -> pd.DataFrame:
    columns = list(dict.fromkeys([
        config.external_sample_id_col, config.external_source_id_col, config.external_group_col,
        *config.external_column_map.values(),
    ]))
    raw = pd.read_csv(config.external_path, usecols=columns)
    frame = pd.DataFrame({
        config.sample_id_col: raw[config.external_sample_id_col].astype(str),
        "external_source_id": raw[config.external_source_id_col].astype(str),
        "external_group_id": raw[config.external_group_col].astype(str),
    })
    for training_column, external_column in config.external_column_map.items():
        frame[training_column] = raw[external_column]
    return frame


def _feature_matrix(frame: pd.DataFrame, feature_set: ExternalFeatureSet) -> np.ndarray:
    matrix, numeric, mask = build_universal_features(
        smiles_cols=list(feature_set.columns),
        numeric_cols=[],
        df=frame,
        desc_name=feature_set.algorithm,
        mode=feature_set.mode,
        descriptor_config=feature_set.params,
    )
    if numeric is not None or not bool(np.asarray(mask, dtype=bool).all()):
        raise FrozenExternalTestError(
            "冻结外部测试描述符不得丢弃行或注入未声明数值特征；请修复输入而不是缩小外部样本"
        )
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise FrozenExternalTestError("描述符产生非有限或空特征矩阵")
    return matrix


def _feature_schema_hash(feature_set: ExternalFeatureSet, train: np.ndarray, external: np.ndarray) -> str:
    if train.shape[1] != external.shape[1]:
        raise FrozenExternalTestError("训练/外部特征维度不同，拒绝预测")
    payload = {
        "feature_set": {"name": feature_set.name, "algorithm": feature_set.algorithm, "columns": list(feature_set.columns), "mode": feature_set.mode, "params": feature_set.params},
        "train_shape": [int(train.shape[0]), int(train.shape[1])],
        "external_shape": [int(external.shape[0]), int(external.shape[1])],
        "dtype": str(train.dtype),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _external_labels_after_predict(config: FrozenExternalTestConfig, expected_ids: Sequence[str]) -> np.ndarray:
    labels = pd.read_csv(
        config.external_path,
        usecols=[config.external_sample_id_col, config.external_label_col],
    )
    ids = _require_unique_text(labels, config.external_sample_id_col, "外部测试标签")
    if set(ids) != set(expected_ids):
        raise FrozenExternalTestError("预测后的外部标签 sample_id 与已预测外部集不一致")
    y = pd.to_numeric(labels[config.external_label_col], errors="coerce")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise FrozenExternalTestError("外部标签含不可解析或非有限值")
    by_id = pd.Series(y.to_numpy(dtype=float), index=ids)
    return by_id.loc[list(expected_ids)].to_numpy(dtype=float)


def _metric_row(contract: FrozenExternalContract, descriptor: str, model: str, y_true: np.ndarray, y_pred: np.ndarray, train_time_s: float, predict_time_s: float) -> Dict[str, Any]:
    tau = kendalltau(y_true, y_pred).statistic
    return {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "evaluation_protocol": EXTERNAL_PROTOCOL,
        "descriptor": descriptor,
        "model": model,
        "n_external": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if np.ptp(y_true) > 0 else float("nan"),
        "kendall_tau": float(tau) if tau is not None else float("nan"),
        "train_time_s": float(train_time_s),
        "predict_time_s": float(predict_time_s),
        "cv_folds": 0,
        "strict_rank_eligible": False,
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render the small external-metric table without optional ``tabulate``.

    The acceptance environment deliberately has only the project's declared
    core dependencies.  Reports are an output contract, so they cannot become
    conditional on pandas' optional ``to_markdown`` extra.
    """
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else "{0:.6f}".format(float(value))
        return str(value)

    headers = [str(column).replace("|", "\\|") for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    return "\n".join(lines)


def _render_reports(contract: FrozenExternalContract, metrics: pd.DataFrame, paths: Mapping[str, Path]) -> None:
    population = contract.population
    table = _markdown_table(metrics)
    markdown = "\n".join([
        "# Frozen development → external test report",
        "",
        "## Protocol boundary",
        "",
        "This report contains one fit on the declared development population and one prediction on the declared external population. It is **not CV**, has no folds, and is ineligible for strict-CV ranking.",
        "",
        "- evaluation protocol: `{0}`".format(EXTERNAL_PROTOCOL),
        "- run ID: `{0}`".format(contract.run_id),
        "- train rows: {0}".format(population["n_train"]),
        "- external rows: {0}".format(population["n_external"]),
        "- external bootstrap groups retained in prediction artifact: {0}".format(population["n_external_groups"]),
        "- train/external sample-ID intersection: 0 (validated before descriptors or fit)",
        "- external labels were loaded only after each prediction was generated in memory.",
        "",
        "## Single external-test metrics",
        "",
        table,
        "",
        "No confidence interval, hypothesis test, or equivalence conclusion is produced here. Those require the pre-specified grouped analysis over the saved external prediction records.",
        "",
        "## Audit fingerprints",
        "",
        "- external standardized data SHA-256: `{0}`".format(contract.config.external_expected_sha256),
        "- audit manifest SHA-256: `{0}`".format(contract.config.audit_manifest_sha256),
        "- train sample-ID SHA-256: `{0}`".format(population["train_sample_id_sha256"]),
        "- external sample-ID SHA-256: `{0}`".format(population["external_sample_id_sha256"]),
    ]) + "\n"
    html_table = metrics.to_html(index=False, float_format=lambda value: "{0:.6f}".format(value))
    html_report = """<!doctype html><html><head><meta charset=\"utf-8\"><title>Frozen external test</title></head><body>
<h1>Frozen development → external test report</h1>
<p><strong>Protocol:</strong> {protocol}. One frozen development fit and one external prediction; not CV and not eligible for strict-CV ranking.</p>
<ul><li>run ID: <code>{run_id}</code></li><li>train rows: {n_train}</li><li>external rows: {n_external}</li><li>external groups: {n_groups}</li><li>train/external sample-ID intersection: 0</li><li>external labels loaded after prediction</li></ul>
<h2>Single external-test metrics</h2>{table}
<p>No interval, hypothesis test, or equivalence conclusion is produced by this single-fit adapter.</p>
</body></html>""".format(
        protocol=html.escape(EXTERNAL_PROTOCOL), run_id=html.escape(contract.run_id),
        n_train=population["n_train"], n_external=population["n_external"],
        n_groups=population["n_external_groups"], table=html_table,
    )
    _atomic_text(markdown, paths["report"] / "external_test_report.md")
    _atomic_text(html_report, paths["report"] / "external_test_report.html")


def run_frozen_external_test(config_path: Path | str) -> int:
    """Execute one frozen-training/external-test task through its YAML contract."""
    config = FrozenExternalTestConfig.from_file(config_path)
    contract = create_frozen_external_contract(config)
    paths = _layout(config.outputs_root)
    manifest_path = paths["manifests"] / "run_manifest.json"
    if config.outputs_root.exists() and not manifest_path.exists() and any(config.outputs_root.iterdir()):
        raise FrozenExternalTestError(
            "输出目录已含未识别文件但没有运行清单；拒绝把单次外部测试混入已有产物：{0}"
            .format(config.outputs_root)
        )
    if manifest_path.exists():
        existing = _read_json(manifest_path, "已有运行清单")
        if existing.get("config_hash") != contract.config_hash:
            raise FrozenExternalTestError("输出目录已被不同配置占用；请新建 result/<task_name>/ 目录")
        raise FrozenExternalTestError("冻结外部测试输出已存在；该适配器不覆盖或重训既有单次外部测试")
    _atomic_json(contract.manifest, manifest_path)
    population_manifest = {
        "run_id": contract.run_id,
        "config_hash": contract.config_hash,
        "evaluation_protocol": EXTERNAL_PROTOCOL,
        "train_population": {
            "path": str(config.dataset_path), "sha256": _sha256_file(config.dataset_path),
            "n_rows": contract.population["n_train"], "sample_id_sha256": contract.population["train_sample_id_sha256"],
        },
        "external_population": {
            "path": str(config.external_path), "sha256": config.external_expected_sha256,
            "audit_manifest": str(config.audit_manifest_path), "audit_manifest_sha256": config.audit_manifest_sha256,
            "n_rows": contract.population["n_external"], "n_groups": contract.population["n_external_groups"],
            "sample_id_sha256": contract.population["external_sample_id_sha256"],
            "source_id_sha256": contract.population["external_source_id_sha256"],
        },
        "sample_id_intersection_count": 0,
        "no_split_manifest": True,
    }
    _atomic_json(population_manifest, paths["manifests"] / "dual_population_manifest.json")

    train = pd.read_csv(config.dataset_path)
    external_features = _canonical_external_feature_frame(config)
    train_ids = train[config.sample_id_col].astype(str).tolist()
    external_ids = external_features[config.sample_id_col].astype(str).tolist()
    y_train = train[config.label_col].to_numpy(dtype=float)
    metric_rows: list[Dict[str, Any]] = []
    for feature_set in config.feature_sets:
        x_train = _feature_matrix(train, feature_set)
        x_external = _feature_matrix(external_features, feature_set)
        schema_hash = _feature_schema_hash(feature_set, x_train, x_external)
        for model in config.models:
            model_config = config.model_configs[model]
            try:
                estimator, factory_audit = construct_estimator(model_config, n_features=x_train.shape[1], n_train=len(x_train))
            except ModelConfigurationError as exc:
                raise FrozenExternalTestError("模型 {0} 无法在外部测试中构造：{1}".format(model, exc)) from exc
            started = datetime.now(timezone.utc).isoformat()
            timer = time.perf_counter()
            estimator.fit(x_train, y_train)
            train_time_s = time.perf_counter() - timer
            timer = time.perf_counter()
            y_pred = np.asarray(estimator.predict(x_external), dtype=float)
            predict_time_s = time.perf_counter() - timer
            if not np.isfinite(y_pred).all():
                raise FrozenExternalTestError("模型 {0} 产生 NaN/inf 外部预测".format(model))
            # This is intentionally the first point at which external labels
            # are read; the estimator has already been fitted and predicted.
            y_true = _external_labels_after_predict(config, external_ids)
            suffix = "{0}__{1}".format(feature_set.name, model)
            prediction = pd.DataFrame({
                "run_id": contract.run_id,
                "config_hash": contract.config_hash,
                "evaluation_protocol": EXTERNAL_PROTOCOL,
                "partition": "external_test",
                "sample_id": external_ids,
                "external_source_id": external_features["external_source_id"].astype(str).tolist(),
                "external_group_id": external_features["external_group_id"].astype(str).tolist(),
                "descriptor": feature_set.name,
                "model": model,
                "y_true": y_true,
                "y_pred": y_pred,
                "feature_schema_hash": schema_hash,
            })
            prediction_path = _atomic_parquet(prediction, paths["predictions"] / (suffix + ".parquet"))
            metadata = {
                "run_id": contract.run_id,
                "config_hash": contract.config_hash,
                "evaluation_protocol": EXTERNAL_PROTOCOL,
                "partition": "frozen_development_to_external_test",
                "descriptor": feature_set.name,
                "model": model,
                "n_train": int(len(train_ids)),
                "n_external": int(len(external_ids)),
                "train_sample_id_sha256": contract.population["train_sample_id_sha256"],
                "external_sample_id_sha256": contract.population["external_sample_id_sha256"],
                "external_source_id_sha256": contract.population["external_source_id_sha256"],
                "external_label_phase": "loaded_after_predict",
                "external_labels_used_for_fit": False,
                "external_labels_used_for_scaling": False,
                "external_labels_used_for_model_selection": False,
                "feature_schema_hash": schema_hash,
                "factory_audit": factory_audit,
                "train_time_s": float(train_time_s),
                "predict_time_s": float(predict_time_s),
                "started_at_utc": started,
                "prediction_path": str(prediction_path),
            }
            _atomic_json(metadata, paths["folds"] / (suffix + ".json"))
            metric_rows.append(_metric_row(contract, feature_set.name, model, y_true, y_pred, train_time_s, predict_time_s))
    metrics = pd.DataFrame.from_records(metric_rows)
    metrics_path = paths["metrics"] / "external_test_metrics.csv"
    if metrics_path.exists():
        raise FrozenExternalTestError("外部测试指标输出已存在，拒绝覆盖：{0}".format(metrics_path))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    _render_reports(contract, metrics, paths)
    print("[frozen-external] run_id={0} train={1} external={2} models={3}".format(
        contract.run_id, len(train_ids), len(external_ids), len(metric_rows)
    ))
    return 0

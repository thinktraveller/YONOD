"""Persistent single-writer Optuna studies outside temporary run staging."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


class StudyStorageError(RuntimeError):
    """A study has conflicting ownership, identity or persisted evidence."""


STORAGE_SCHEMA = "yonod-hpo-study/v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def study_identity(payload: Mapping[str, Any]) -> str:
    """Hash caller supplied *content* identities, never absolute task paths."""
    required = {
        "purpose", "feature_identity", "training_ids", "training_labels",
        "inner_folds", "preprocessing", "fixed_model_config", "search_space",
        "objective", "seed", "budget", "versions",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise StudyStorageError("study 身份缺少字段：" + ", ".join(missing))
    if payload["purpose"] not in {"outer_fold", "final_model"}:
        raise StudyStorageError("study purpose 必须是 outer_fold 或 final_model")
    if payload["purpose"] == "outer_fold" and not isinstance(payload.get("outer_fold"), Mapping):
        raise StudyStorageError("外层折 study 必须记录 repeat/fold 身份")
    if "output_dir" in payload or "config_path" in payload:
        raise StudyStorageError("study 内容身份不得包含绝对输出路径")
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:24]


def dependency_versions(model: str) -> dict[str, str]:
    """Record exact packages that can change suggestions or model results."""
    names = ["optuna", "numpy", "scikit-learn", "pandas"]
    if model == "xgb":
        names.append("xgboost")
    elif model == "lightgbm":
        names.append("lightgbm")
    elif model != "rf":
        raise StudyStorageError(f"不支持的 HPO 模型：{model}")
    return {name: importlib.metadata.version(name) for name in names}


def source_fingerprints(modules: Mapping[str, Path]) -> dict[str, str]:
    """Bind search semantics to local code bytes without hashing path names."""
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in modules.items()}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".hpo-", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _single_writer(path: Path) -> Iterator[None]:
    """Cross-platform, process-bound lock; exiting releases a stale writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"\0")
        stream.flush()
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise StudyStorageError(f"study 已有活跃写者：{path}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _trial_snapshot(trial: Any) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
    }


class PersistentStudy:
    """One isolated study at outputs.root/hpo/<combination>/<study_id>."""

    def __init__(self, output_root: Path, combination: str, identity: Mapping[str, Any]) -> None:
        if not combination or combination in {".", ".."} or Path(combination).name != combination:
            raise StudyStorageError("组合标识必须是单个安全目录名")
        self.payload = dict(identity)
        self.study_id = study_identity(self.payload)
        self.root = Path(output_root).resolve() / "hpo" / combination / self.study_id
        self.manifest_path = self.root / "study_manifest.json"
        self.trials_path = self.root / "trials.json"
        self.database_path = self.root / "study.sqlite"
        self.lock_path = self.root / ".writer.lock"

    @contextmanager
    def open(self) -> Iterator[Any]:
        """Own one study, recover interrupted trials, then expose its handle."""
        import optuna
        from sqlalchemy.engine import URL

        self.root.mkdir(parents=True, exist_ok=True)
        with _single_writer(self.lock_path):
            expected = {"schema": STORAGE_SCHEMA, "study_id": self.study_id, "identity": self.payload}
            if self.manifest_path.exists():
                try:
                    actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise StudyStorageError("study manifest 无法读取") from exc
                if actual != expected:
                    raise StudyStorageError("study manifest 身份冲突，拒绝复用")
            elif self.database_path.exists():
                raise StudyStorageError("已有数据库缺少身份 manifest，拒绝猜测恢复")
            else:
                _atomic_json(self.manifest_path, expected)
            storage = optuna.storages.RDBStorage(url=str(URL.create("sqlite", database=str(self.database_path))))
            study = optuna.create_study(
                study_name=self.study_id, storage=storage, load_if_exists=True,
                direction=self.payload["objective"]["direction"],
                sampler=optuna.samplers.TPESampler(seed=self.payload["seed"]),
                pruner=optuna.pruners.NopPruner(),
            )
            for trial in study.get_trials(deepcopy=False):
                if trial.state == optuna.trial.TrialState.RUNNING:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            self._reconcile(study)
            try:
                yield study
            finally:
                self._reconcile(study)

    def _reconcile(self, study: Any) -> None:
        """Derived JSON may be rebuilt after DB commit; conflict is corruption."""
        expected = [_trial_snapshot(trial) for trial in study.get_trials(deepcopy=False)]
        if self.trials_path.exists():
            try:
                existing = json.loads(self.trials_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise StudyStorageError("trial 导出损坏") from exc
            if not isinstance(existing, list) or len(existing) > len(expected) or existing != expected[:len(existing)]:
                raise StudyStorageError("trial 导出与数据库不一致，拒绝覆盖证据")
        _atomic_json(self.trials_path, expected)

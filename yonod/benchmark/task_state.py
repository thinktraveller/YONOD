"""SQLite-backed, single-process task state for resumable benchmarks.

The executor writes one prediction shard per external fold.  This module owns
the *small* state database that answers whether that fold is still pending,
currently running, reusable, or needs a deliberate rerun.  It never stores
feature matrices or predictions in SQLite.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence


class TaskStateError(RuntimeError):
    """Raised when a task state transition would make recovery ambiguous."""


TERMINAL_STATES = {"succeeded", "failed", "interrupted", "skipped"}
VALID_STATES = {"pending", "running", *TERMINAL_STATES}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    """The immutable identity of one ``descriptor × model × repeat × fold`` job."""

    run_id: str
    config_hash: str
    split_id: str
    descriptor: str
    model: str
    repeat: int
    fold: int

    @property
    def task_key(self) -> str:
        return "{0}__{1}__{2}__{3}__r{4:02d}__f{5:02d}".format(
            self.run_id,
            self.split_id,
            self.descriptor,
            self.model,
            self.repeat,
            self.fold,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_key": self.task_key,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "split_id": self.split_id,
            "descriptor": self.descriptor,
            "model": self.model,
            "repeat": int(self.repeat),
            "fold": int(self.fold),
        }


@dataclass(frozen=True)
class ClaimedTask:
    """A task atomically claimed by this process and ready for execution."""

    spec: TaskSpec
    attempts: int
    started_at_utc: str


def collect_output_record(
    prediction_path: Path | str,
    metadata_path: Path | str,
    expected: TaskSpec,
) -> Dict[str, Any]:
    """Return file hashes after proving a fold's metadata belongs to ``expected``.

    The prediction Parquet itself is intentionally not loaded here: this state
    layer verifies file identity and metadata ownership, while the executor is
    responsible for the prediction-schema and row-level manifest checks.
    """
    prediction = Path(prediction_path).resolve()
    metadata = Path(metadata_path).resolve()
    if not prediction.is_file() or not metadata.is_file():
        raise TaskStateError("成功任务缺少预测分片或折级元数据")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskStateError("折级元数据无法读取：{0}".format(exc)) from exc
    for field, value in {
        "run_id": expected.run_id,
        "config_hash": expected.config_hash,
        "split_id": expected.split_id,
        "descriptor": expected.descriptor,
        "model": expected.model,
        "repeat": expected.repeat,
        "fold": expected.fold,
    }.items():
        if payload.get(field) != value:
            raise TaskStateError("折级元数据字段 {0!r} 与任务键不一致".format(field))
    return {
        "prediction_path": str(prediction),
        "metadata_path": str(metadata),
        "prediction_sha256": _sha256_file(prediction),
        "metadata_sha256": _sha256_file(metadata),
    }


def verify_output_record(record: Mapping[str, Any], expected: TaskSpec) -> Optional[str]:
    """Return ``None`` only when the persisted output is still reusable."""
    try:
        actual = collect_output_record(
            Path(str(record["prediction_path"])),
            Path(str(record["metadata_path"])),
            expected,
        )
    except (KeyError, TaskStateError) as exc:
        return str(exc)
    for field in ("prediction_sha256", "metadata_sha256"):
        if actual[field] != record.get(field):
            return "输出哈希不匹配：{0}".format(field)
    return None


class TaskStateStore:
    """Transactional state store for the serial benchmark coordinator.

    SQLite ``BEGIN IMMEDIATE`` protects every claim/transition.  This is a
    correctness-first single-process implementation; no worker should write
    the same database concurrently until a separate coordinator is introduced.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    split_id TEXT NOT NULL,
                    descriptor TEXT NOT NULL,
                    model TEXT NOT NULL,
                    repeat_index INTEGER NOT NULL,
                    fold INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('pending','running','succeeded','failed','interrupted','skipped')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at_utc TEXT NOT NULL,
                    started_at_utc TEXT,
                    heartbeat_at_utc TEXT,
                    completed_at_utc TEXT,
                    worker_info_json TEXT,
                    output_record_json TEXT,
                    error_type TEXT,
                    error_summary TEXT,
                    log_path TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_key TEXT NOT NULL REFERENCES tasks(task_key),
                    at_utc TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    message TEXT
                )
                """
            )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        task_key: str,
        from_status: Optional[str],
        to_status: str,
        message: Optional[str] = None,
    ) -> None:
        connection.execute(
            "INSERT INTO task_events(task_key, at_utc, from_status, to_status, message) VALUES (?, ?, ?, ?, ?)",
            (task_key, _utc_now(), from_status, to_status, message),
        )

    def sync_tasks(self, specs: Iterable[TaskSpec]) -> int:
        """Insert planned tasks and reject an incompatible identity reuse."""
        count = 0
        with self._transaction() as connection:
            for spec in specs:
                existing = connection.execute(
                    "SELECT * FROM tasks WHERE task_key = ?", (spec.task_key,)
                ).fetchone()
                if existing is not None:
                    for field, value in spec.as_dict().items():
                        db_field = "repeat_index" if field == "repeat" else field
                        if field != "task_key" and existing[db_field] != value:
                            raise TaskStateError("任务键被不兼容的配置复用：{0}".format(spec.task_key))
                    continue
                payload = spec.as_dict()
                connection.execute(
                    """INSERT INTO tasks(
                        task_key, run_id, config_hash, split_id, descriptor, model,
                        repeat_index, fold, status, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (
                        payload["task_key"], payload["run_id"], payload["config_hash"],
                        payload["split_id"], payload["descriptor"], payload["model"],
                        payload["repeat"], payload["fold"], _utc_now(),
                    ),
                )
                self._event(connection, spec.task_key, None, "pending", "任务已登记")
                count += 1
        return count

    def audit_succeeded_outputs(self) -> int:
        """Requeue succeeded tasks whose files are missing, altered or mismatched."""
        requeued = 0
        with self._transaction() as connection:
            rows = connection.execute("SELECT * FROM tasks WHERE status = 'succeeded'").fetchall()
            for row in rows:
                spec = TaskSpec(
                    row["run_id"], row["config_hash"], row["split_id"], row["descriptor"],
                    row["model"], int(row["repeat_index"]), int(row["fold"]),
                )
                try:
                    record = json.loads(row["output_record_json"] or "{}")
                except json.JSONDecodeError:
                    record = {}
                reason = verify_output_record(record, spec)
                if reason is None:
                    continue
                message = "成功输出审计失败，已重新排队：{0}".format(reason)
                connection.execute(
                    """UPDATE tasks SET status='pending', completed_at_utc=NULL,
                       output_record_json=NULL, error_type='OutputIntegrityError',
                       error_summary=? WHERE task_key=?""",
                    (message, spec.task_key),
                )
                self._event(connection, spec.task_key, "succeeded", "pending", message)
                requeued += 1
        return requeued

    def recover_stale_running(self, stale_after_seconds: float) -> int:
        """Mark abandoned workers interrupted; they are rerunnable with ``rerun_failed``."""
        if stale_after_seconds < 0:
            raise TaskStateError("stale_after_seconds 不能为负数")
        cutoff = datetime.now(timezone.utc).timestamp() - stale_after_seconds
        changed = 0
        with self._transaction() as connection:
            rows = connection.execute("SELECT task_key, heartbeat_at_utc FROM tasks WHERE status='running'").fetchall()
            for row in rows:
                heartbeat = row["heartbeat_at_utc"]
                try:
                    timestamp = datetime.fromisoformat(heartbeat).timestamp() if heartbeat else float("-inf")
                except ValueError:
                    timestamp = float("-inf")
                if timestamp > cutoff:
                    continue
                message = "运行心跳超时，标记为 interrupted，等待明确重跑"
                connection.execute(
                    """UPDATE tasks SET status='interrupted', completed_at_utc=?,
                       error_type='InterruptedTask', error_summary=? WHERE task_key=?""",
                    (_utc_now(), message, row["task_key"]),
                )
                self._event(connection, row["task_key"], "running", "interrupted", message)
                changed += 1
        return changed

    def claim_next(
        self,
        *,
        rerun_failed: bool = False,
        task_keys: Optional[Sequence[str]] = None,
    ) -> Optional[ClaimedTask]:
        """Atomically claim one eligible task, or return ``None`` when none remain."""
        allowed = ["pending"]
        if rerun_failed:
            allowed.extend(["failed", "interrupted"])
        placeholders = ",".join("?" for _ in allowed)
        where = "status IN ({0})".format(placeholders)
        values: list[Any] = list(allowed)
        if task_keys is not None:
            if not task_keys:
                return None
            where += " AND task_key IN ({0})".format(",".join("?" for _ in task_keys))
            values.extend(task_keys)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE {0} ORDER BY repeat_index, fold, descriptor, model, task_key LIMIT 1".format(where),
                values,
            ).fetchone()
            if row is None:
                return None
            started = _utc_now()
            worker = json.dumps(
                {"hostname": socket.gethostname(), "platform": platform.platform(), "pid": None},
                ensure_ascii=False,
                sort_keys=True,
            )
            connection.execute(
                """UPDATE tasks SET status='running', attempts=attempts+1,
                   started_at_utc=?, heartbeat_at_utc=?, completed_at_utc=NULL,
                   worker_info_json=?, error_type=NULL, error_summary=NULL, log_path=NULL
                   WHERE task_key=?""",
                (started, started, worker, row["task_key"]),
            )
            self._event(connection, row["task_key"], row["status"], "running", "任务已被单进程协调器认领")
            spec = TaskSpec(
                row["run_id"], row["config_hash"], row["split_id"], row["descriptor"],
                row["model"], int(row["repeat_index"]), int(row["fold"]),
            )
            return ClaimedTask(spec, int(row["attempts"]) + 1, started)

    def heartbeat(self, task_key: str) -> None:
        with self._transaction() as connection:
            result = connection.execute(
                "UPDATE tasks SET heartbeat_at_utc=? WHERE task_key=? AND status='running'",
                (_utc_now(), task_key),
            )
            if result.rowcount != 1:
                raise TaskStateError("只能为 running 任务更新心跳：{0}".format(task_key))

    def mark_succeeded(
        self,
        claimed: ClaimedTask,
        prediction_path: Path | str,
        metadata_path: Path | str,
    ) -> None:
        record = collect_output_record(prediction_path, metadata_path, claimed.spec)
        with self._transaction() as connection:
            row = connection.execute("SELECT status FROM tasks WHERE task_key=?", (claimed.spec.task_key,)).fetchone()
            if row is None or row["status"] != "running":
                raise TaskStateError("只能将当前 running 任务标记为 succeeded")
            connection.execute(
                """UPDATE tasks SET status='succeeded', completed_at_utc=?, heartbeat_at_utc=?,
                   output_record_json=?, error_type=NULL, error_summary=NULL WHERE task_key=?""",
                (_utc_now(), _utc_now(), json.dumps(record, ensure_ascii=False, sort_keys=True), claimed.spec.task_key),
            )
            self._event(connection, claimed.spec.task_key, "running", "succeeded", "预测分片及元数据哈希校验通过")

    def mark_failed(self, claimed: ClaimedTask, error: BaseException, log_path: Optional[Path | str] = None) -> None:
        summary = "{0}: {1}".format(type(error).__name__, str(error)).strip()[:2000]
        with self._transaction() as connection:
            row = connection.execute("SELECT status FROM tasks WHERE task_key=?", (claimed.spec.task_key,)).fetchone()
            if row is None or row["status"] != "running":
                raise TaskStateError("只能将当前 running 任务标记为 failed")
            connection.execute(
                """UPDATE tasks SET status='failed', completed_at_utc=?, heartbeat_at_utc=?,
                   error_type=?, error_summary=?, log_path=? WHERE task_key=?""",
                (_utc_now(), _utc_now(), type(error).__name__, summary, str(log_path) if log_path else None, claimed.spec.task_key),
            )
            self._event(connection, claimed.spec.task_key, "running", "failed", summary)

    def summary(self) -> Dict[str, int]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        counts = {state: 0 for state in VALID_STATES}
        counts.update({str(row["status"]): int(row["n"]) for row in rows})
        counts["total"] = sum(counts[state] for state in VALID_STATES)
        return counts

    def get_task(self, task_key: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_key=?", (task_key,)).fetchone()
        if row is None:
            raise KeyError(task_key)
        return dict(row)

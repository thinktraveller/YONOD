"""Persistent active-time budget for a single, serial modeling task.

The ledger checkpoints active monotonic time every half second.  After an
unclean exit, only time through the last durable checkpoint is charged; the
wall-clock gap is recorded separately and never silently resets the budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .storage import StudyStorageError, _atomic_json, _canonical, _single_writer


class TaskBudgetExpired(RuntimeError):
    """The soft task deadline prevents starting another operation."""


class ActiveBudgetLedger:
    """One writer owns the task budget while modeling is active."""

    SCHEMA = "yonod-hpo-task-budget/v1"

    def __init__(
        self, output_root: Path, identity: Mapping[str, Any], timeout_s: float | None,
        *, study_dir: Path | None = None,
    ) -> None:
        if timeout_s is not None and (not math.isfinite(float(timeout_s)) or timeout_s <= 0):
            raise ValueError("task_timeout_s 必须为有限正数")
        self.root = Path(study_dir).resolve() if study_dir is not None else Path(output_root).resolve() / "hpo"
        self.path = self.root / ("study_budget.json" if study_dir is not None else "task_budget.json")
        self.lock_path = self.root / (".study_budget.lock" if study_dir is not None else ".task_budget.lock")
        self.identity = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer = None
        self._started_monotonic = 0.0
        self._started_charged = 0.0
        self._state: dict[str, Any] | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> "ActiveBudgetLedger":
        self.root.mkdir(parents=True, exist_ok=True)
        self._writer = _single_writer(self.lock_path)
        self._writer.__enter__()
        try:
            if self.path.exists():
                try:
                    state = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise StudyStorageError("任务时间账本无法读取") from exc
                if (state.get("schema"), state.get("identity"), state.get("timeout_s")) != (
                    self.SCHEMA, self.identity, self.timeout_s,
                ):
                    raise StudyStorageError("任务时间账本身份或时限冲突")
                if not isinstance(state.get("charged_s"), (int, float)) or state["charged_s"] < 0:
                    raise StudyStorageError("任务时间账本累计耗时无效")
                if state.get("active"):
                    state["interrupted_sessions"] = int(state.get("interrupted_sessions", 0)) + 1
                    state["downtime_s"] = float(state.get("downtime_s", 0.0)) + max(
                        0.0, time.time() - float(state["last_checkpoint_wall_s"])
                    )
            else:
                state = {
                    "schema": self.SCHEMA, "identity": self.identity,
                    "timeout_s": self.timeout_s, "charged_s": 0.0,
                    "downtime_s": 0.0, "interrupted_sessions": 0,
                    "sessions": 0, "active": False,
                }
            self._state = state
            self._started_charged = float(state["charged_s"])
            self._started_monotonic = time.monotonic()
            state["sessions"] = int(state.get("sessions", 0)) + 1
            state["active"] = True
            self._checkpoint()
            self._thread = threading.Thread(target=self._heartbeat, name="yonod-hpo-budget", daemon=True)
            self._thread.start()
            return self
        except BaseException:
            self._writer.__exit__(None, None, None)
            self._writer = None
            raise

    def _checkpoint(self) -> None:
        with self._lock:
            assert self._state is not None
            self._state["charged_s"] = self._started_charged + max(0.0, time.monotonic() - self._started_monotonic)
            self._state["last_checkpoint_wall_s"] = time.time()
            _atomic_json(self.path, self._state)

    def _heartbeat(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._checkpoint()
            except BaseException as exc:
                self._error = exc
                self._stop.set()

    def elapsed_s(self) -> float:
        if self._error is not None:
            raise StudyStorageError("任务时间账本写入失败") from self._error
        return self._started_charged + max(0.0, time.monotonic() - self._started_monotonic)

    def remaining_s(self) -> float | None:
        if self.timeout_s is None:
            return None
        return max(0.0, self.timeout_s - self.elapsed_s())

    def require_start(self, operation: str) -> None:
        remaining = self.remaining_s()
        if remaining is not None and remaining <= 0:
            raise TaskBudgetExpired(f"HPO task_timeout_s 已耗尽，不能开始{operation}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            if self._state is not None:
                self._state["active"] = False
                self._checkpoint()
            if self._error is not None:
                raise StudyStorageError("任务时间账本写入失败") from self._error
        finally:
            if self._writer is not None:
                self._writer.__exit__(exc_type, exc, traceback)
                self._writer = None

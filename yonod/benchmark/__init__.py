"""YONOD benchmark pipeline 的可审计运行基础设施。"""

from .config import (
    BenchmarkConfig,
    BenchmarkContract,
    BenchmarkConfigError,
    create_benchmark_contract,
    write_run_manifest,
)
from .executor import FoldExecutionError, FoldExecutionResult, execute_fold
from .task_state import (
    ClaimedTask,
    TaskSpec,
    TaskStateError,
    TaskStateStore,
    collect_output_record,
    verify_output_record,
)
from .metrics import (
    COMBINATION_TIME_SUMMARY_COLUMNS,
    DIMENSION_TIME_SUMMARY_COLUMNS,
    MetricRebuildError,
    MetricRebuildResult,
    paired_comparisons,
    rebuild_fold_metrics,
    summarize_combination_times,
    summarize_dimension_times,
    summarize_combinations,
    tukey_hsd_comparisons,
    write_combination_time_summary,
    write_dimension_time_summaries,
    write_metric_tables,
)
from .report import BenchmarkReportError, BenchmarkReportResult, generate_benchmark_report

__all__ = [
    "BenchmarkConfig",
    "BenchmarkContract",
    "BenchmarkConfigError",
    "create_benchmark_contract",
    "write_run_manifest",
    "FoldExecutionError",
    "FoldExecutionResult",
    "execute_fold",
    "ClaimedTask",
    "TaskSpec",
    "TaskStateError",
    "TaskStateStore",
    "collect_output_record",
    "verify_output_record",
    "COMBINATION_TIME_SUMMARY_COLUMNS",
    "DIMENSION_TIME_SUMMARY_COLUMNS",
    "MetricRebuildError",
    "MetricRebuildResult",
    "paired_comparisons",
    "rebuild_fold_metrics",
    "summarize_combination_times",
    "summarize_dimension_times",
    "summarize_combinations",
    "tukey_hsd_comparisons",
    "write_combination_time_summary",
    "write_dimension_time_summaries",
    "write_metric_tables",
    "BenchmarkReportError",
    "BenchmarkReportResult",
    "generate_benchmark_report",
]

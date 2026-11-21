"""Shared metric, configuration and time descriptions for human-readable reports."""

from collections.abc import Mapping
from html import escape
import json
from math import isfinite


_METRICS = (
    ("R²（决定系数）", r"R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}",
     "用参考产率的平均值作为常量预测，和模型的平方误差总和比较。1 表示预测与参考值完全一致，0 表示与常量预测相当，负值表示模型的平方误差更大。参考产率完全相同时，R² 无定义。"),
    ("RMSE（均方根误差）", r"\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(y_i-\hat y_i)^2}",
     "把每条预测误差平方，取平均后开平方。数值越小，预测越接近参考值。平方会放大大误差的影响。"),
    ("MAE（平均绝对误差）", r"\mathrm{MAE}=\frac{1}{n}\sum_i\lvert y_i-\hat y_i\rvert",
     "把每条预测误差取绝对值后平均，表示预测与参考值平均相差多少。数值越小，预测越接近参考值。"),
)
_SYMBOLS = "公式中，yᵢ 是参考产率，ŷᵢ 是预测产率，ȳ 是本次评估所用参考产率的平均值，n 是纳入计算的记录数。"
_UNITS = "R² 无单位。RMSE 和 MAE 与参考产率使用相同标度。产率按 0 到 1 记录时，误差 0.10 对应 10 个百分点。"
_HTML_FORMULAS = (
    "R² = 1 − 模型平方误差之和 / 常量预测的平方误差之和",
    "RMSE = √(平方误差之和 / 记录数)",
    "MAE = 绝对误差之和 / 记录数",
)
_DDOF_HTML = (
    '<math aria-label="ddof 等于零"><semantics>'
    '<mrow><mi mathvariant="normal">ddof</mi><mo>=</mo><mn>0</mn></mrow>'
    r'<annotation encoding="application/x-tex">\mathrm{ddof}=0</annotation></semantics></math>'
)


def population_sd_note(*, html: bool = False) -> str:
    """Explain what is measured before introducing the notation."""
    notation = _DDOF_HTML if html else r"$\mathrm{ddof}=0$"
    return (
        "总体标准差衡量各个数值与均值的偏离程度。将每个偏差平方后取平均，再开平方。"
        f"计算平方偏差的平均值时，以纳入统计的数值个数为分母（{notation}）。"
    )


def metric_explanation_markdown() -> str:
    paragraphs = [_SYMBOLS]
    for name, formula, description in _METRICS:
        paragraphs.extend((f"**{name}**。{description}", f"$${formula}$$"))
    paragraphs.append(_UNITS)
    return "\n\n".join(paragraphs)


def metric_explanation_html() -> str:
    paragraphs = [f"<p>{escape(_SYMBOLS)}</p>"]
    for (name, _, description), formula in zip(_METRICS, _HTML_FORMULAS):
        paragraphs.extend((f"<p><strong>{escape(name)}</strong>。{escape(description)}</p>",
                           f'<p class="formula-block">{escape(formula)}</p>'))
    paragraphs.append(f"<p>{escape(_UNITS)}</p>")
    return "\n".join(paragraphs)


def ranking_formula(weights, *, html=False) -> str:
    terms = (
        f"{weights['r2']:g} × R²",
        f"{weights['rmse']:g} × (1 − RMSE / 最大 RMSE)",
        f"{weights['mae']:g} × (1 − MAE / 最大 MAE)",
    )
    formula = "综合分 = " + " + ".join(terms)
    note = "误差最大值取自本次有排名资格的组合，计算时分母下限为 10⁻⁹。权重由 outputs.ranking.weights 配置。"
    if html:
        return f'<p class="formula-block">{escape(formula)}</p><p class="note">{escape(note)}</p>'
    return f"`{formula}`\n\n{note}"


def _display_setting(value):
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def evaluation_rows(config):
    """Read reportable evaluation settings without serializing audit internals."""
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping) or not evaluation:
        return []
    fields = (
        ("评估协议", "protocol"),
        ("每轮折数", "n_splits"),
        ("重复轮数", "n_repeats"),
        ("划分前打乱", "shuffle"),
        ("种子", "seed"),
        ("固定划分名单", "split_manifest"),
    )
    rows = [(label, _display_setting(evaluation[key])) for label, key in fields
            if evaluation.get(key) is not None]
    for block, keys in (
        ("grouping", (("分组规则", "strategy"), ("分组列", "group_column"))),
        ("external_test", (("外测数据集", "path"), ("外测参考值列", "label_col"),
                           ("外测分组列", "group_col"), ("外测输入列映射", "column_map"))),
    ):
        values = evaluation.get(block)
        if isinstance(values, Mapping):
            rows.extend((label, _display_setting(values[key])) for label, key in keys
                        if values.get(key) is not None)
    return rows


def chart_time_unit(seconds):
    """Return the multiplier and unit for seconds displayed on one common axis."""
    if seconds <= 0:
        return 1, "秒", "s"
    if seconds < 0.001:
        return 1e6, "微秒", "µs"
    if seconds < 1:
        return 1e3, "毫秒", "ms"
    if seconds < 60:
        return 1, "秒", "s"
    if seconds < 3600:
        return 1 / 60, "分钟", "min"
    return 1 / 3600, "小时", "h"


def format_duration(value):
    """Format finite nonnegative seconds with a unit matching their scale."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "—"
    if not isfinite(seconds) or seconds < 0:
        return "—"
    multiplier, _, unit = chart_time_unit(seconds)
    return f"{seconds * multiplier:.4g} {unit}"

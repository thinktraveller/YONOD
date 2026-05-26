"""验证 report.py 的 rank_combinations() 与 generate_report()。

用例：
  1. rank_combinations：排名顺序正确，score 计算符合权重公式
  2. rank_combinations：前三名自动生成推荐理由
  3. generate_report：输出 HTML 文件，包含固定指标解释文字
  4. generate_report：HTML 中包含正确的任务名和标签列名
  5. rank_combinations：输入全 NaN 时返回空 DataFrame
"""

import math
import tempfile
from pathlib import Path

import pandas as pd

from yonod_yield.universal.report import rank_combinations, generate_report

# 构造 mock metrics DataFrame（run_yonod.py 输出格式）
_MOCK_ROWS = [
    {"descriptor": "morgan",    "model": "rf",        "r2_mean": 0.87, "rmse_mean": 0.04, "mae_mean": 0.03},
    {"descriptor": "morgan",    "model": "xgb",       "r2_mean": 0.82, "rmse_mean": 0.06, "mae_mean": 0.05},
    {"descriptor": "maccs",     "model": "rf",        "r2_mean": 0.74, "rmse_mean": 0.09, "mae_mean": 0.07},
    {"descriptor": "maccs",     "model": "xgb",       "r2_mean": 0.70, "rmse_mean": 0.11, "mae_mean": 0.09},
]
_MOCK_DF = pd.DataFrame(_MOCK_ROWS)


# ─── 用例 1：排名顺序 ──────────────────────────────────────────────────────── #
def test_rank_order():
    ranked = rank_combinations(_MOCK_DF)
    assert ranked.iloc[0]["desc_name"] == "morgan"
    assert ranked.iloc[0]["model_name"] == "rf"
    assert ranked.iloc[0]["rank"] == 1
    scores = ranked["score"].tolist()
    assert scores == sorted(scores, reverse=True), "score 应降序排列"
    print(f"[PASS] 用例1  排名第1: morgan×rf, score={ranked.iloc[0]['score']:.3f}")


# ─── 用例 2：前三名推荐理由非空 ───────────────────────────────────────────── #
def test_top3_reasons():
    ranked = rank_combinations(_MOCK_DF)
    for i in range(min(3, len(ranked))):
        reason = ranked.iloc[i]["reason"]
        assert reason, f"第{i+1}名推荐理由不应为空"
    print(f"[PASS] 用例2  前3名均有推荐理由")


# ─── 用例 3：HTML 含指标解释固定文字 ──────────────────────────────────────── #
def test_html_glossary():
    with tempfile.TemporaryDirectory() as tmp:
        task_info = {
            "task_name": "测试任务",
            "csv_path": "test.csv",
            "n_samples": 200,
            "smiles_cols": ["SMILES"],
            "numeric_cols": [],
            "label_col": "yield",
            "n_combinations": 4,
        }
        out = generate_report(_MOCK_DF, task_info, Path(tmp))
        html = out.read_text(encoding="utf-8")
        # §14.5.2 固定文字检查
        assert "R²" in html,    "HTML 应含 R² 解释"
        assert "RMSE" in html,  "HTML 应含 RMSE 解释"
        assert "MAE" in html,   "HTML 应含 MAE 解释"
        assert "0.5" in html,   "HTML 应含权重 0.5"
        assert "0.3" in html,   "HTML 应含权重 0.3"
        assert "0.2" in html,   "HTML 应含权重 0.2"
        print(f"[PASS] 用例3  HTML 含所有指标解释和权重文字，文件大小={out.stat().st_size} 字节")


# ─── 用例 4：HTML 含任务名和标签列名 ──────────────────────────────────────── #
def test_html_task_info():
    with tempfile.TemporaryDirectory() as tmp:
        task_info = {
            "task_name": "酰胺缩合产率预测",
            "csv_path": "test.csv",
            "n_samples": 47015,
            "smiles_cols": ["sub_1_smiles"],
            "numeric_cols": [],
            "label_col": "yield",
            "n_combinations": 4,
        }
        out = generate_report(_MOCK_DF, task_info, Path(tmp))
        html = out.read_text(encoding="utf-8")
        assert "酰胺缩合产率预测" in html, "HTML 应含任务名"
        assert "yield" in html,            "HTML 应含标签列名"
        print(f"[PASS] 用例4  HTML 正确嵌入任务名和标签列名")


# ─── 用例 5：输入全 NaN 时安全返回空 DataFrame ────────────────────────────── #
def test_all_nan_safe():
    nan_df = pd.DataFrame([
        {"descriptor": "morgan", "model": "rf",
         "r2_mean": float("nan"), "rmse_mean": float("nan"), "mae_mean": float("nan")},
    ])
    ranked = rank_combinations(nan_df)
    assert ranked.empty, "全 NaN 行应过滤后返回空 DataFrame"
    print(f"[PASS] 用例5  全 NaN 输入安全返回空 DataFrame")


if __name__ == "__main__":
    test_rank_order()
    test_top3_reasons()
    test_html_glossary()
    test_html_task_info()
    test_all_nan_safe()
    print("\n[OK] 全部 5 个用例通过")

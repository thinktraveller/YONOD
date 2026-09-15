"""Keep strict benchmark report plotting compatible with supported Matplotlibs."""

from __future__ import annotations

import unittest

from yonod.benchmark.report import _boxplot_with_label_compat


class _NewAxes:
    def __init__(self) -> None:
        self.calls = []

    def boxplot(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return "new"


class _OldAxes:
    def __init__(self) -> None:
        self.calls = []

    def boxplot(self, data, **kwargs):
        self.calls.append((data, kwargs))
        if "tick_labels" in kwargs:
            raise TypeError("boxplot() got an unexpected keyword argument 'tick_labels'")
        return "old"


class _BrokenAxes:
    def boxplot(self, data, **kwargs):
        raise TypeError("data cannot be converted to float")


class BenchmarkReportMatplotlibCompatibilityTests(unittest.TestCase):
    def test_prefers_current_tick_labels_keyword(self) -> None:
        axes = _NewAxes()
        self.assertEqual(_boxplot_with_label_compat(axes, [[1.0]], ["rf"]), "new")
        self.assertEqual(axes.calls[0][1]["tick_labels"], ["rf"])
        self.assertNotIn("labels", axes.calls[0][1])

    def test_falls_back_only_for_legacy_tick_labels_typeerror(self) -> None:
        axes = _OldAxes()
        self.assertEqual(_boxplot_with_label_compat(axes, [[1.0]], ["rf"]), "old")
        self.assertEqual(len(axes.calls), 2)
        self.assertEqual(axes.calls[1][1]["labels"], ["rf"])

    def test_does_not_hide_unrelated_plotting_typeerror(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot be converted"):
            _boxplot_with_label_compat(_BrokenAxes(), [[1.0]], ["rf"])


if __name__ == "__main__":
    unittest.main()

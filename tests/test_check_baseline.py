import json
import sys
import tempfile
import unittest
from pathlib import Path

# ``scripts/`` is not a package; import the comparator by file path.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import check_baseline  # noqa: E402


class _BaseGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


class MemoryEvalGateTests(_BaseGateTest):
    def test_passes_when_recall_unchanged(self) -> None:
        baseline = {"summary": {"recall_at_k": {"1": 0.5, "5": 0.75}, "mrr": 0.625}}
        report = {"summary": {"recall_at_k": {"1": 0.5, "5": 0.75}, "mrr": 0.625}}
        baseline_path = self._write("baseline.json", baseline)
        report_path = self._write("report.json", report)
        rc = check_baseline.main([
            "memory-eval",
            "--report", str(report_path),
            "--baseline", str(baseline_path),
        ])
        self.assertEqual(rc, 0)

    def test_fails_when_recall_drops_past_threshold(self) -> None:
        baseline = {"summary": {"recall_at_k": {"1": 0.50, "5": 0.75}, "mrr": 0.625}}
        report = {"summary": {"recall_at_k": {"1": 0.30, "5": 0.50}, "mrr": 0.40}}
        baseline_path = self._write("baseline.json", baseline)
        report_path = self._write("report.json", report)
        rc = check_baseline.main([
            "memory-eval",
            "--report", str(report_path),
            "--baseline", str(baseline_path),
            "--max-drop-pp", "5.0",
        ])
        self.assertEqual(rc, 1)


class BenchToolsGateTests(_BaseGateTest):
    def test_unverified_baseline_passes_with_warning(self) -> None:
        baseline = {
            "status": "unverified",
            "rounds": 0,
            "counts": {"tool_calls": 0, "tool_request": 0, "both": 0, "neither": 0},
            "canonical_path_hint": "unverified",
        }
        report = baseline
        baseline_path = self._write("baseline.json", baseline)
        report_path = self._write("report.json", report)
        rc = check_baseline.main([
            "bench-tools",
            "--report", str(report_path),
            "--baseline", str(baseline_path),
        ])
        self.assertEqual(rc, 0)

    def test_canonical_drift_fails(self) -> None:
        baseline = {
            "rounds": 5,
            "counts": {"tool_calls": 4, "tool_request": 0, "both": 1, "neither": 0},
            "canonical_path_hint": "tool_calls",
        }
        report = {
            "rounds": 5,
            "counts": {"tool_calls": 0, "tool_request": 4, "both": 1, "neither": 0},
            "canonical_path_hint": "tool_request",
        }
        baseline_path = self._write("baseline.json", baseline)
        report_path = self._write("report.json", report)
        rc = check_baseline.main([
            "bench-tools",
            "--report", str(report_path),
            "--baseline", str(baseline_path),
        ])
        self.assertEqual(rc, 1)

    def test_too_many_neither_rounds_fails(self) -> None:
        baseline = {
            "rounds": 5,
            "counts": {"tool_calls": 4, "tool_request": 0, "both": 1, "neither": 0},
            "canonical_path_hint": "tool_calls",
        }
        report = {
            "rounds": 4,
            "counts": {"tool_calls": 1, "tool_request": 0, "both": 0, "neither": 3},
            "canonical_path_hint": "tool_calls",
        }
        baseline_path = self._write("baseline.json", baseline)
        report_path = self._write("report.json", report)
        rc = check_baseline.main([
            "bench-tools",
            "--report", str(report_path),
            "--baseline", str(baseline_path),
            "--allow-neither-pct", "20",
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

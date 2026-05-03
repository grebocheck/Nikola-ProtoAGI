"""Portable baseline comparator for ProtoAGI regression gates.

Drives two gates today:

- ``memory-eval``: compares ``recall@k`` and ``mrr`` against
  ``runs/memory-eval-baseline.json``.
- ``bench-tools``: compares the dominant ``canonical_path_hint`` (and the
  count of the ``neither`` bucket) against
  ``runs/bench-tools-baseline.json``.

Both gates are invoked the same way:

    python scripts/check_baseline.py memory-eval --report runs/<latest>.json
    python scripts/check_baseline.py bench-tools --report runs/<latest>.json

Exit code 0 means within tolerance, 1 means a regression. The script never
runs the underlying eval — call ``protoagi memory-eval --json --output …``
or ``protoagi bench-tools --output …`` first and pass the resulting file in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINES = {
    "memory-eval": PROJECT_ROOT / "runs" / "memory-eval-baseline.json",
    "bench-tools": PROJECT_ROOT / "runs" / "bench-tools-baseline.json",
}


class GateError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON in {path}: {exc}") from exc


def _check_memory_eval(report_path: Path, baseline_path: Path, max_drop_pp: float) -> int:
    report = _load_json(report_path)
    baseline = _load_json(baseline_path)
    current = (report.get("summary") or {}).get("recall_at_k") or {}
    base = (baseline.get("summary") or {}).get("recall_at_k") or {}
    if not current or not base:
        raise GateError("recall_at_k missing from report or baseline")
    threshold = max_drop_pp / 100.0
    failed = False
    for key in sorted(set(current) | set(base)):
        cur = float(current.get(key, 0.0))
        ref = float(base.get(key, 0.0))
        delta = cur - ref
        sign = "+" if delta >= 0 else ""
        print(
            f"recall@{key}: current={cur:.3f} baseline={ref:.3f} delta={sign}{delta:.3f}"
        )
        if delta < -threshold:
            print(f"  -> drop > {max_drop_pp}pp", file=sys.stderr)
            failed = True
    cur_mrr = float((report.get("summary") or {}).get("mrr") or 0.0)
    ref_mrr = float((baseline.get("summary") or {}).get("mrr") or 0.0)
    delta_mrr = cur_mrr - ref_mrr
    sign_mrr = "+" if delta_mrr >= 0 else ""
    print(f"MRR:        current={cur_mrr:.3f} baseline={ref_mrr:.3f} delta={sign_mrr}{delta_mrr:.3f}")
    if delta_mrr < -threshold:
        print(f"  -> MRR drop > {max_drop_pp}pp", file=sys.stderr)
        failed = True
    return 1 if failed else 0


def _check_bench_tools(report_path: Path, baseline_path: Path, allow_neither_pct: float) -> int:
    report = _load_json(report_path)
    baseline = _load_json(baseline_path)
    if baseline.get("status") == "unverified":
        print(
            "bench-tools baseline marked 'unverified' — capture one with "
            "`protoagi bench-tools --output runs/bench-tools-baseline.json` "
            "against the production model, then re-run the gate.",
            file=sys.stderr,
        )
        return 0
    cur_canon = str(report.get("canonical_path_hint") or "unverified")
    ref_canon = str(baseline.get("canonical_path_hint") or "unverified")
    cur_counts = dict(report.get("counts") or {})
    rounds = max(1, int(report.get("rounds") or 1))
    neither_pct = (cur_counts.get("neither", 0) * 100.0) / rounds
    print(f"canonical_path_hint: current={cur_canon} baseline={ref_canon}")
    print(f"counts: {cur_counts}")
    print(f"neither_pct={neither_pct:.1f}% (allowed <= {allow_neither_pct:.1f}%)")
    failed = False
    if cur_canon != ref_canon:
        print("  -> canonical path drifted", file=sys.stderr)
        failed = True
    if neither_pct > allow_neither_pct:
        print("  -> too many 'neither' rounds", file=sys.stderr)
        failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a fresh report against a stored baseline.")
    sub = parser.add_subparsers(dest="gate", required=True)

    memory_eval = sub.add_parser("memory-eval", help="memory-eval recall@k gate")
    memory_eval.add_argument("--report", required=True)
    memory_eval.add_argument("--baseline", default=str(DEFAULT_BASELINES["memory-eval"]))
    memory_eval.add_argument("--max-drop-pp", type=float, default=5.0)

    bench_tools = sub.add_parser("bench-tools", help="bench-tools canonical-path gate")
    bench_tools.add_argument("--report", required=True)
    bench_tools.add_argument("--baseline", default=str(DEFAULT_BASELINES["bench-tools"]))
    bench_tools.add_argument(
        "--allow-neither-pct",
        type=float,
        default=20.0,
        help="Maximum share of rounds that may classify as 'neither' before the gate fails.",
    )

    args = parser.parse_args(argv)
    try:
        if args.gate == "memory-eval":
            return _check_memory_eval(
                Path(args.report).resolve(),
                Path(args.baseline).resolve(),
                args.max_drop_pp,
            )
        if args.gate == "bench-tools":
            return _check_bench_tools(
                Path(args.report).resolve(),
                Path(args.baseline).resolve(),
                args.allow_neither_pct,
            )
    except GateError as exc:
        print(f"gate error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

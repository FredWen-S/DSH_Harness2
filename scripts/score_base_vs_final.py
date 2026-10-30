# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score each task's first base rollout and its selected final with the same judge.

The paper's "Base harness" baseline is the unmodified harness on the same task; the shared
base executions of a Harness² run are exactly that, so `base1` is scored as the baseline
and the selected final as Harness². Run after `harness2` has finished the namespace.

Every scored run must carry a verdict from the configured judge (`HARNESS2_JUDGE_MODEL`);
a verdict cached under another judge is set aside and re-scored by the adapter, and the
script refuses to write a comparison that mixes judges. Rollouts whose DSH stream touched
ground-truth paths are reported, not silently scored.

    python scripts/score_base_vs_final.py --bench lab --substrate dsh --namespace lab-dsh-agg1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from harness2.benchmarks.lab import judge_id
from harness2.cli import _common as C
from harness2.substrate import dsh as DSH


def _frac(s: dict | None) -> str:
    if not s or not s.get("n_criteria"):
        return "unscored"
    return f"{s['n_passed']}/{s['n_criteria']} ({s['n_passed'] / s['n_criteria']:.1%})"


def _scored(adapter, run_dir: Path, task: str, judges: set[str]) -> dict | None:
    """Score one run and record which judge produced the verdict and what it touched."""
    s = adapter.score(run_dir, task)
    verdict = json.loads((run_dir / "scores.json").read_text(encoding="utf-8")) if s else {}
    judges.add(verdict.get("judge_model", "?"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {**(s or {}), "forbidden_access": metrics.get("forbidden_access", [])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default="lab")
    p.add_argument("--substrate", default="dsh")
    p.add_argument("--namespace", required=True, help="e.g. lab-dsh-agg1")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    adapter = C.make_adapter(args.bench, None, args.substrate)
    expected = judge_id(adapter.judge_model)
    root = adapter.spec.runs_root / "adapt" / args.namespace
    rows, judges = [], set()
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        task = summary["task_id"]
        base = next((b for b in summary.get("base") or [] if b["label"] == "base1"), None)
        row = {"task_id": task, "base": None, "finals": {}}
        if base:
            row["base"] = _scored(adapter, Path(base["run_dir"]), task, judges)
            row["base_usage"] = DSH.token_usage(Path(base["run_dir"]) / DSH.STREAM_NAME)
        for mode, sel in (summary.get("selections") or {}).items():
            run_dir = sel.get("chosen_run_dir")
            row["finals"][mode] = {
                "label": sel.get("chosen_label"),
                "tree_is_base": sel.get("chosen_tree_is_base"),
                "score": _scored(adapter, Path(run_dir), task, judges) if run_dir else None,
                "lesson": (sel.get("experience") or {}).get("lesson", ""),
            }
        rows.append(row)
        finals = "; ".join(
            f"{m}: {_frac(f['score'])} [{f['label']}{' = base tree' if f['tree_is_base'] else ''}]"
            for m, f in row["finals"].items()
        )
        print(f"{task}\n  base1: {_frac(row['base'])}\n  final: {finals}", flush=True)
    if judges - {expected}:
        print(f"refusing to compare: verdicts from {sorted(judges)}, expected {expected!r}")
        return 1
    usage = {"rollouts": 0, "input": 0, "output": 0, "cache_read": 0}
    for stream in (adapter._results_root / args.namespace).rglob(DSH.STREAM_NAME):
        u = DSH.token_usage(stream)
        usage["rollouts"] += 1
        for k in ("input", "output", "cache_read"):
            usage[k] += u[k]
    report = {
        "judge": adapter.judge_model,
        "judge_endpoint": os.environ.get("HARNESS2_JUDGE_BASE_URL", "default"),
        "rows": rows,
        "rollout_usage": usage,
    }
    out = args.out or root / "base_vs_final.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"rollout tokens: {usage}\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

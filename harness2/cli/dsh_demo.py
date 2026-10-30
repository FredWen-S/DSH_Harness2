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

"""Run the full method live on the DSH substrate, then score base rollouts against finals.

Every rollout and improver session is a real DSH headless turn; scoring happens only after
the stream has finished and selection has committed, exactly as `harness2-judge` would.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.proposer import Proposer
from ..core.schedule import mode_token, run_domain
from ..dsh_demo import DshDemoAdapter
from ..substrate import dsh as DSH


def _score_task(adapter: DshDemoAdapter, summary: dict) -> dict:
    task_id = summary["task_id"]
    row = {"task_id": task_id, "base": [], "finals": {}}
    for b in summary.get("base") or []:
        s = adapter.score(Path(b["run_dir"]), task_id)
        row["base"].append({"label": b["label"], **(s or {})})
    for mode, sel in (summary.get("selections") or {}).items():
        run_dir = sel.get("chosen_run_dir")
        s = adapter.score(Path(run_dir), task_id) if run_dir else None
        row["finals"][mode] = {
            "label": sel.get("chosen_label"),
            "is_base": sel.get("chosen_tree_is_base"),
            "lesson": (sel.get("experience") or {}).get("lesson", ""),
            **(s or {"pass": False, "got": "(no final)"}),
        }
    return row


def _usage(root: Path) -> dict:
    tot = {"rollouts": 0, "input": 0, "output": 0, "cache_read": 0}
    for p in (root / "executions").rglob(DSH.STREAM_NAME):
        u = DSH.token_usage(p)
        tot["rollouts"] += 1
        for k in ("input", "output", "cache_read"):
            tot[k] += u[k]
    return tot


def _markdown(rows: list[dict], modes: tuple[str, ...]) -> str:
    head = "| task | base pass | " + " | ".join(f"{m} final" for m in modes) + " |"
    lines = [head, "|" + "---|" * (2 + len(modes))]
    for r in rows:
        base = sum(bool(b.get("pass")) for b in r["base"])
        cells = [f"{base}/{len(r['base'])}"]
        for m in modes:
            f = r["finals"].get(m) or {}
            mark = "pass" if f.get("pass") else "fail"
            cells.append(f"{mark} ({f.get('label')}, got {f.get('got')})")
        lines.append(f"| {r['task_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/dsh-demo"))
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--mode", choices=("parallel", "sequential", "both"), default="parallel")
    parser.add_argument("--tasks", nargs="*", help="task ids to run (default: all, in order)")
    parser.add_argument("--model", default=None, help="solver provider/model for rollouts")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")
    root = args.output.expanduser().resolve()
    adapter = DshDemoAdapter(root, model=args.model)
    adapter.substrate.preflight()
    tasks = args.tasks or adapter.tasks()
    proposer = Proposer(adapter.spec.channels["proposer"], audit_dir=root / "audit")
    modes = ("agg", "seq") if args.mode == "both" else (mode_token(args.mode),)
    result = run_domain(
        "bookkeeping",
        tasks,
        args.k,
        adapter=adapter,
        prop=proposer,
        runs_root=adapter.spec.runs_root,
        library_root=root / "library" / f"k{args.k}",
        modes=modes,
    )
    rows = []
    for summary_path in sorted(adapter.spec.runs_root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("task_id") in tasks:
            rows.append(_score_task(adapter, summary))
    order = {t: i for i, t in enumerate(tasks)}
    rows.sort(key=lambda r: order.get(r["task_id"], 99))
    report = {
        "k": args.k,
        "modes": list(modes),
        "solver": adapter.model,
        "improver": adapter.spec.channels["proposer"].model,
        "stream": result,
        "scores": rows,
        "usage": _usage(root),
        "improver_sessions": len(proposer._calls),
    }
    (root / "dsh_demo.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = _markdown(rows, modes)
    (root / "dsh_demo.md").write_text(table, encoding="utf-8")
    print(table)
    print(
        f"DSH demo: {result['done']} tasks completed; {result['failed']} failed; "
        f"{report['usage']['rollouts']} rollouts, {report['improver_sessions']} improver "
        f"sessions. {root / 'dsh_demo.json'}"
    )
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Pool `base_vs_final.json` over seeds: paired base1-vs-final criterion fractions.

Each (task, seed) is one pair: the unmodified harness and the selected Harness² final on
the same task, scored by the same judge. Reports the per-pair differences, their mean with
a percentile-bootstrap 95% interval, an exact two-sided sign test, any ground-truth access,
and rollout token totals priced at the given per-1M rates.

    python scripts/aggregate_seeds.py --namespaces lab-dsh-agg1-s1 lab-dsh-agg1-s2 lab-dsh-agg1-s3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

from harness2 import config


def _frac(s: dict | None) -> float | None:
    if not s or not s.get("n_criteria"):
        return None
    return s["n_passed"] / s["n_criteria"]


def _sign_test(diffs: list[float]) -> float:
    """Exact two-sided sign test p-value, ties dropped."""
    pos = sum(d > 0 for d in diffs)
    n = sum(d != 0 for d in diffs)
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def _bootstrap(diffs: list[float], reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespaces", nargs="+", required=True)
    p.add_argument(
        "--price",
        nargs=3,
        type=float,
        default=(0.66, 0.022, 1.98),
        metavar=("IN", "CACHE", "OUT"),
        help="$ per 1M input / cache-read / output",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    pairs, usage, judges, flagged = [], {"input": 0, "cache_read": 0, "output": 0}, set(), []
    for ns in args.namespaces:
        path = config.RUNS / "lab" / "adapt" / ns / "base_vs_final.json"
        if not path.is_file():
            print(f"missing {path}")
            return 1
        data = json.loads(path.read_text(encoding="utf-8"))
        judges.add(data.get("judge"))
        for k in usage:
            usage[k] += data["rollout_usage"][k]
        for row in data["rows"]:
            final = row["finals"].get("agg") or {}
            b, f = _frac(row["base"]), _frac(final.get("score"))
            for arm, s in (("base1", row["base"]), ("final", final.get("score"))):
                if s and s.get("forbidden_access"):
                    flagged.append((ns, row["task_id"], arm, s["forbidden_access"]))
            pairs.append(
                {
                    "namespace": ns,
                    "task": row["task_id"].split("/", 1)[1],
                    "base": b,
                    "final": f,
                    "final_is_base_tree": final.get("tree_is_base"),
                    "lesson": final.get("lesson", ""),
                }
            )
    if len(judges) != 1:
        print(f"refusing to pool across judges: {judges}")
        return 1
    done = [x for x in pairs if x["base"] is not None and x["final"] is not None]
    diffs = [x["final"] - x["base"] for x in done]
    print(f"{'task':58s} {'ns':18s} base   final  diff")
    for x in done:
        tag = " (base tree)" if x["final_is_base_tree"] else ""
        print(
            f"{x['task'][:58]:58s} {x['namespace'][-6:]:18s} {x['base']:.3f}  "
            f"{x['final']:.3f}  {x['final'] - x['base']:+.3f}{tag}"
        )
    summary = {"judge": judges.pop(), "pairs": pairs, "n_scored": len(done)}
    if diffs:
        lo, hi = _bootstrap(diffs)
        mean = sum(diffs) / len(diffs)
        summary.update(
            mean_diff=mean,
            ci95=[lo, hi],
            wins=sum(d > 0 for d in diffs),
            losses=sum(d < 0 for d in diffs),
            ties=sum(d == 0 for d in diffs),
            sign_test_p=_sign_test(diffs),
            mean_base=sum(x["base"] for x in done) / len(done),
            mean_final=sum(x["final"] for x in done) / len(done),
        )
        print(
            f"\nn={len(diffs)}  base {summary['mean_base']:.3f}  final "
            f"{summary['mean_final']:.3f}  mean diff {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
            f"W/L/T {summary['wins']}/{summary['losses']}/{summary['ties']}  "
            f"sign p={summary['sign_test_p']:.3f}"
        )
    a, c, o = args.price
    summary["rollout_usage"] = usage
    summary["rollout_cost_usd"] = (
        usage["input"] * a + usage["cache_read"] * c + usage["output"] * o
    ) / 1e6
    summary["forbidden_access"] = flagged
    print(
        f"rollout tokens {usage}  ~${summary['rollout_cost_usd']:.2f}  "
        f"forbidden access: {flagged or 'none'}"
    )
    out = args.out or config.RUNS / "lab" / "adapt" / "aggregate.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

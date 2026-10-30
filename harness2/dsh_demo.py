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

"""A live, model-backed adapter for exercising the method on the DSH substrate.

Three bookkeeping tasks in one domain. Each workspace holds a ledger and a NOTES.md stating
that workspace's conventions (accounting-notation negatives, voided rows, European number
format); the task prompt points at the conventions without restating them. Rollouts and
improver sessions are real DSH headless turns. The expected totals live only in this module:
the improver never sees them, and `score` is called only after selection.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from decimal import Decimal
from pathlib import Path

from . import config
from .core import runmark as RM
from .core import tree
from .core.spec import BenchPrompts, BenchSpec, ChannelSpec
from .substrate import dsh as DSH
from .substrate.base import Install

TASKS: dict[str, dict] = {
    "ledger-notation": {
        "ledger.csv": (
            "date,description,amount,status\n"
            '2026-09-01,Client retainer,"$1,200.50",POSTED\n'
            "2026-09-03,Refund to client,(300.00),POSTED\n"
            "2026-09-05,Bank fee,-45.10,POSTED\n"
            "2026-09-09,Interest,15,POSTED\n"
        ),
        "NOTES.md": (
            "# Bookkeeping conventions\n\n"
            "- Amounts in parentheses are negative (accounting notation): (300.00) is -300.00.\n"
            "- Amounts may carry a currency symbol and thousands separators.\n"
        ),
        "expected": "870.40",
    },
    "ledger-void": {
        "ledger.csv": (
            "date,description,amount,status\n"
            '2026-09-02,Invoice 1182,"2,000.00",POSTED\n'
            "2026-09-04,Supplier credit,(150.25),POSTED\n"
            "2026-09-06,Duplicate invoice 1182,999.99,VOID\n"
            "2026-09-08,Office supplies,$80.00,POSTED\n"
            "2026-09-10,Reversed fee,(20.00),VOID\n"
        ),
        "NOTES.md": (
            "# Bookkeeping conventions\n\n"
            "- Rows whose status is VOID were reversed and must not be counted.\n"
            "- Amounts in parentheses are negative.\n"
        ),
        "expected": "1929.75",
    },
    "ledger-european": {
        "ledger.csv": (
            "datum;omschrijving;bedrag;status\n"
            "2026-09-01;Factuur 77;1.234,56;POSTED\n"
            "2026-09-02;Creditnota;(100,00);POSTED\n"
            "2026-09-03;Bankkosten;-34,56;POSTED\n"
            "2026-09-04;Geannuleerd;500,00;VOID\n"
        ),
        "NOTES.md": (
            "# Bookkeeping conventions\n\n"
            "- This ledger is semicolon-separated and uses European number format: '.' groups "
            "thousands and ',' is the decimal mark.\n"
            "- Rows whose status is VOID must not be counted; parentheses mean negative.\n"
        ),
        "expected": "1100.00",
    },
}

TASK_PROMPT = (
    "Compute the net total of the `{column}` column in the ledger in this workspace, applying "
    "the bookkeeping conventions that apply here. Write exactly one line, `Total: <amount>` "
    "with two decimal places and no currency symbol, to output/answer.md."
)

BASE_SYSTEM = (
    "You are a bookkeeping assistant working inside a task workspace. Do the task with the "
    "tools available and save the requested output where the task says.\n"
)

# The module holding the expected totals and the reports derived from them.
_FORBIDDEN = ("dsh_demo.py", "dsh_demo.json", "dsh_demo.md", "scores.json")

_TOTAL = re.compile(r"Total:\s*(-?[0-9]+\.[0-9]{2})\s*$")


class DshDemoAdapter:
    """The BenchmarkAdapter surface for the live ledger tasks, on the DSH substrate."""

    def __init__(self, root: Path, *, model: str | None = None, timeout: int = 900):
        self.root = Path(root)
        self.model = model or DSH.SOLVER_MODEL
        self.timeout = timeout
        self.substrate = DSH.DshSubstrate()
        channel = ChannelSpec("dsh", config.DSH_IMPROVER_MODEL, steps=60, timeout=1800)
        self.spec = BenchSpec(
            name="dshdemo",
            runs_root=self.root / "runs",
            install_mode=DSH.INSTALL_MODE,
            prompts=BenchPrompts(
                work=(
                    "compute the net total of a small transaction ledger, honouring the "
                    "workspace's own bookkeeping conventions, and write it in the requested "
                    "one-line format"
                ),
                domain_word="bookkeeping area",
                deliverable_word="answer file",
            ),
            channels={"proposer": channel, "selector": channel},
            domain_word="bookkeeping area",
        )
        self.execution_count = 0
        os.environ[DSH.FORBIDDEN_ENV] = os.pathsep.join(DSH.forbidden_needles(_FORBIDDEN))
        base = {name: "" for name in self.substrate.surface().always_on}
        base["systemprompt.md"] = BASE_SYSTEM
        tree.write_tree(self.substrate.surface(), base, self.base_harness(), stubs=False)
        for task_id, body in TASKS.items():
            src = self.task_files(task_id)
            src.mkdir(parents=True, exist_ok=True)
            for name in ("ledger.csv", "NOTES.md"):
                (src / name).write_text(body[name], encoding="utf-8")

    def tasks(self, split: str = "all") -> list[str]:
        return list(TASKS)

    def domain_of(self, task_id: str) -> str:
        return "bookkeeping"

    def task_prompt(self, task_id: str) -> str:
        column = "bedrag" if task_id == "ledger-european" else "amount"
        return TASK_PROMPT.format(column=column)

    def task_files(self, task_id: str) -> Path:
        return self.root / "sources" / task_id

    def base_harness(self) -> Path:
        return self.root / "base"

    def validate_tree(self, harness: tree.Tree) -> None:
        problems = DSH.dsh_tree_problems(harness)
        if problems:
            raise ValueError("; ".join(problems))

    def _install(self, harness_dir: Path | None) -> Install:
        return self.substrate.materialise(harness_dir or self.base_harness(), self.spec, {})

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        run_dir = self.root / "executions" / run_id
        stream = run_dir / DSH.STREAM_NAME
        # Reuse only a finished run of this exact harness (the marker fingerprints its tree).
        if RM.is_complete(run_dir, harness_dir=harness_dir) and self.delivered(run_dir):
            return run_dir
        shutil.rmtree(run_dir, ignore_errors=True)
        workspace = run_dir / "workspace"
        shutil.copytree(self.task_files(task_id), workspace)
        (workspace / "output").mkdir()
        install = self._install(harness_dir)
        try:
            DSH.install_scripts(install, workspace)
            env = self.substrate.run_env(os.environ, install, self.model)
            self.execution_count += 1
            rc, wall, tail = DSH.run_headless(
                self.task_prompt(task_id),
                workspace,
                env,
                stream,
                timeout=int(timeout or self.timeout),
            )
            touched = DSH.forbidden_touches(stream, DSH.forbidden_needles(_FORBIDDEN))
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "rc": rc,
                        "wall_s": wall,
                        "stderr_tail": tail,
                        "forbidden_access": touched,
                        **DSH.token_usage(stream),
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.rmtree(install.staging_dir, ignore_errors=True)
        if DSH.completed(stream):
            RM.mark_complete(run_dir, run_id=run_id, harness_dir=harness_dir)
        return run_dir

    def _answer(self, run_dir: Path) -> Path:
        return Path(run_dir) / "workspace" / "output" / "answer.md"

    def delivered(self, run_dir: Path) -> bool:
        p = self._answer(run_dir)
        return p.is_file() and bool(p.read_text(encoding="utf-8", errors="replace").strip())

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        return self.substrate.read_trajectory(
            Path(run_dir) / DSH.STREAM_NAME, obs_trunc=obs_trunc, max_chars=10**9
        )

    def deliverable_text(self, run_dir: Path) -> str:
        p = self._answer(run_dir)
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return (
            {"output/answer.md": self.deliverable_text(run_dir)} if self.delivered(run_dir) else {}
        )

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        text = self.deliverable_text(run_dir).strip()
        m = _TOTAL.match(text.splitlines()[-1]) if text else None
        got = m.group(1) if m else ""
        expected = TASKS[task_id]["expected"]
        ok = bool(got) and Decimal(got) == Decimal(expected)
        return {"pass": ok, "got": got or text[:80], "expected": expected}


__all__ = ["DshDemoAdapter", "TASKS"]

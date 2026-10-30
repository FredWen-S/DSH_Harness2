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

"""LAB (harvey-labs) adapter: one class for both substrate cells.

Upstream tools run in subprocesses under `config.LAB_PYTHON`.
`_task_json` strips grading criteria before exposing task data to the improver.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import replace
from functools import cache
from pathlib import Path

from .. import config
from .._compat import SIGKILL, kill_group
from ..core import runmark as RM
from ..core import tree as HT
from ..core.spec import BenchPrompts, BenchSpec, resolve_channel_model
from ..substrate import dsh as DSH
from ..substrate.base import Install, RunRequest
from ..substrate.codex import CODEX_SURFACE, CodexSubstrate
from ..substrate.dsh import DshSubstrate
from ..substrate.opencode import OPENCODE_SURFACE, OpencodeSubstrate
from .tasks import discover, select

JUDGE_MODEL = config.LAB_JUDGE_MODEL

_BASE_LOCK = threading.Lock()
_BASE_DONE: set[Path] = set()

LAB_BASE = BenchPrompts(
    work=(
        "Professional legal work for a supervising lawyer: analysing matter records, reviewing "
        "and drafting instruments, and preparing research or advice in the exact files requested. "
        "The source documents and instructions control. The result must be matter-ready: complete, "
        "internally consistent, traceable to the record, and usable by another lawyer without "
        "repair."
    ),
    domain_word="practice area",
    deliverable_word="legal deliverable",
    inv_levers=(
        "- inventory the matter files before choosing a reading order; map each requested output "
        "and issue to the sources likely to control it\n"
        "- use the environment's documented binary-document extraction path, preserve pinpoint "
        "locations, and distinguish unread material from material that says nothing\n"
        "- establish an output plan early, write every exact requested filename before the budget "
        "is at risk, then reopen each artifact and check that it parses\n"
        "- reconcile names, dates, amounts, defined terms and cross-document conflicts with a "
        "recorded source hierarchy instead of silently choosing one version"
    ),
    dom_levers=(
        "- identify the correct legal work product and the audience-dependent structure it needs\n"
        "- separate source text, matter fact, legal inference, risk assessment and recommendation; "
        "do not present one as another\n"
        "- test parties, authority, jurisdiction, effective dates, conditions, exceptions, timing, "
        "remedies, definitions and cross-references where they matter to the assignment\n"
        "- verify authorities and jurisdiction-sensitive propositions from available primary "
        "material or an authorised research path; never manufacture a citation or rule"
    ),
    yardstick=(
        "Did the run identify the controlling matter record, perform the legal operation the "
        "instructions called for, preserve traceability for important propositions, produce every "
        "named file in usable form, and inspect the finished artifact before stopping?"
    ),
    selection_checks=(
        "Prefer the work product that uses the correct instrument or memo form; covers every "
        "requested issue and output; states matter facts, calculations and quoted provisions "
        "consistently with the supplied record; distinguishes uncertainty from conclusion; gives "
        "actionable analysis rather than document summary; and is internally coherent enough for "
        "a lawyer to review rather than reconstruct."
    ),
)

LAB_WORK_TYPES: dict[str, str] = {
    "analyze": (
        "WORK TYPE — ANALYSE. Build an issue inventory before writing. For each material issue, "
        "connect the governing source language and facts to the consequence, severity or decision, "
        "then state a practical recommendation. Reconcile related documents and calculations; a "
        "long summary that never performs the requested analysis is not complete."
    ),
    "review": (
        "WORK TYPE — REVIEW. Establish the baseline document and the proposed or competing text. "
        "Identify each consequential change or defect, explain its legal and operational effect, "
        "and recommend accept, reject or revise with precise replacement language when requested. "
        "Preserve the requested markup, comparison and output conventions."
    ),
    "draft": (
        "WORK TYPE — DRAFT. Confirm the correct instrument, parties, authority and intended legal "
        "effect before drafting. Check definitions, operative provisions, conditions, exceptions, "
        "dates, remedies, exhibits, cross-references, signature blocks and execution formalities "
        "as applicable. Every clause must fit the matter rather than read like generic filler."
    ),
    "research": (
        "WORK TYPE — RESEARCH. Define the question, jurisdiction, relevant date and source "
        "hierarchy first. Prefer primary authority, verify currency through the available research "
        "path, distinguish binding from persuasive material, apply the authorities to the matter "
        "facts, and expose contrary authority, gaps and assumptions."
    ),
}

LAB_PRACTICE_AREAS: dict[str, str] = {
    "antitrust-competition": (
        "PRACTICE LENS — ANTITRUST / COMPETITION. Define the relevant conduct, parties, market and "
        "jurisdictions before applying any framework. Track filing or process timing, theories of "
        "harm, evidence for market power and practical remedy or clearance implications separately."
    ),
    "arbitration-international-dispute-resolution": (
        "PRACTICE LENS — ARBITRATION. Keep the arbitration agreement, governing law, seat, rules, "
        "tribunal authority, procedural timetable and enforcement posture distinct. Tie every "
        "procedural recommendation to the controlling instrument or rule supplied for the matter."
    ),
    "banking-finance": (
        "PRACTICE LENS — BANKING / FINANCE. Map parties, facilities, commitments, conditions, "
        "security, priority, covenants, representations, draw mechanics, payment terms, defaults "
        "and remedies. Reconcile defined terms and amounts across the full document set."
    ),
    "bankruptcy-restructuring": (
        "PRACTICE LENS — RESTRUCTURING. Track the entity, claim or contract, procedural posture, "
        "priority, collateral, stay or avoidance implications, deadlines and required approvals. "
        "Separate legal entitlement from likely recovery and implementation risk."
    ),
    "capital-markets": (
        "PRACTICE LENS — CAPITAL MARKETS. Reconcile issuer, security, offering structure, "
        "disclosure, eligibility, approvals, closing conditions and distribution restrictions. "
        "Check that risk factors and disclosure are specific to the supplied facts and consistent "
        "across documents."
    ),
    "contracts": (
        "PRACTICE LENS — CONTRACTS. Build a clause map covering parties, scope, obligations, "
        "pricing, conditions, change control, term, termination, liability, indemnity, "
        "confidentiality, assignment, notices and dispute terms as relevant. Follow definitions "
        "and exceptions before stating an obligation or remedy."
    ),
    "corporate-governance": (
        "PRACTICE LENS — CORPORATE GOVERNANCE. Confirm entity type, governing documents, "
        "decision-making body, authority, quorum, voting threshold, conflicts, approvals, records "
        "and filing steps. Do not assign an act to a board, holder or officer without checking "
        "authority."
    ),
    "corporate-ma": (
        "PRACTICE LENS — M&A. Connect diligence findings to transaction mechanics: parties, "
        "structure, signing and closing sequence, approvals, consents, conditions, covenants, "
        "purchase-price effects, risk allocation and post-closing obligations. Quantify exposure "
        "from the record."
    ),
    "data-privacy-cybersecurity": (
        "PRACTICE LENS — PRIVACY / CYBERSECURITY. Map data, actors, systems, jurisdictions, "
        "purpose, authority, transfer, retention, security controls and incident duties. Separate "
        "contractual, regulatory and operational requirements and avoid asserting a regime without "
        "confirming scope."
    ),
    "diligence": (
        "PRACTICE LENS — DILIGENCE. Use a complete document and issue inventory, record what was "
        "and was not reviewed, trace findings to exact sources, distinguish absence of evidence "
        "from a clean finding, rank materiality consistently and connect each risk to a next "
        "action."
    ),
    "emerging-companies-venture-capital": (
        "PRACTICE LENS — EMERGING COMPANIES / VENTURE. Reconcile charter, capitalization, "
        "securities, investor rights, board and holder approvals, vesting, protective provisions "
        "and financing documents. Test economics and governance together rather than reviewing "
        "forms in isolation."
    ),
    "employment-labor": (
        "PRACTICE LENS — EMPLOYMENT / LABOR. Identify worker population, jurisdiction, status, "
        "policy and timeline. Separate contractual promises, statutory duties and company "
        "practice; handle protected information carefully and make recommendations operationally "
        "implementable."
    ),
    "energy-natural-resources": (
        "PRACTICE LENS — ENERGY / NATURAL RESOURCES. Map assets, rights, operators, jurisdictions, "
        "permits, production or supply obligations, environmental duties, pricing, measurement, "
        "transport and decommissioning risk. Reconcile technical schedules with operative terms."
    ),
    "environmental-esg": (
        "PRACTICE LENS — ENVIRONMENTAL / ESG. Identify facility or activity, jurisdiction, permit "
        "or standard, reporting period, source data and responsible party. Distinguish legal "
        "compliance, contractual allocation and voluntary representation; verify metrics and "
        "boundaries."
    ),
    "funds-asset-management": (
        "PRACTICE LENS — FUNDS / ASSET MANAGEMENT. Reconcile vehicle, manager, investors, mandate, "
        "economics, allocations, conflicts, valuation, liquidity, governance, side arrangements "
        "and reporting. Test consistency across governing and subscription materials."
    ),
    "healthcare-life-sciences": (
        "PRACTICE LENS — HEALTHCARE / LIFE SCIENCES. Identify product or service, actors, data, "
        "jurisdiction, approval or reimbursement pathway, promotional or research limits and "
        "patient impact. Verify specialised requirements from supplied or authorised sources."
    ),
    "immigration": (
        "PRACTICE LENS — IMMIGRATION. Build a person-specific chronology and status record, "
        "identify the requested classification or benefit, eligibility, documentary proof, filing "
        "sequence, dependencies and deadlines. Never fill an evidentiary gap with assumption."
    ),
    "insurance": (
        "PRACTICE LENS — INSURANCE. Map insured, policy, period, coverage grant, definitions, "
        "limits, retentions, conditions, exclusions, endorsements, notice and claimed loss. "
        "Analyse coverage and factual causation separately before stating the likely outcome."
    ),
    "intellectual-property": (
        "PRACTICE LENS — INTELLECTUAL PROPERTY. Identify the asset, ownership chain, scope, "
        "territory, term, registration or evidence, licences, restrictions, improvements, "
        "enforcement and third-party rights. Keep different IP regimes and chain-of-title "
        "questions distinct."
    ),
    "international-trade-sanctions": (
        "PRACTICE LENS — TRADE / SANCTIONS. Establish parties, ownership and control, goods or "
        "services, route, jurisdictions, relevant date, licences and end use. Verify current "
        "restrictions from authorised sources and state screening assumptions rather than guessing."
    ),
    "litigation-dispute-resolution": (
        "PRACTICE LENS — LITIGATION. Build claims, elements, defenses, burden, evidence and "
        "procedural posture from the record. Maintain a reliable chronology and citation trail; "
        "connect each factual proposition to proof and each requested action to the applicable "
        "procedure."
    ),
    "real-estate": (
        "PRACTICE LENS — REAL ESTATE. Reconcile parties, property description, title, interests, "
        "diligence, financing, conditions, adjustments, closing, possession, use restrictions and "
        "post-closing duties. Confirm dates and defined property across every instrument."
    ),
    "structured-finance-securitization": (
        "PRACTICE LENS — STRUCTURED FINANCE. Map assets, originator, transfer chain, vehicle, "
        "waterfall, credit support, triggers, eligibility, servicing, representations and "
        "reporting. Reconcile defined calculations and priority rules across transaction documents."
    ),
    "tax": (
        "PRACTICE LENS — TAX. Fix taxpayer, entity, jurisdiction, transaction, tax period and "
        "factual assumptions first. Show computational steps, distinguish character, timing and "
        "reporting, and verify current law or rates from authorised materials instead of memory."
    ),
    "trusts-estates-private-client": (
        "PRACTICE LENS — TRUSTS / ESTATES / PRIVATE CLIENT. Map family and fiduciary "
        "relationships, assets, governing instruments, capacity, authority, dispositive plan, "
        "conditions, tax and administration. Check names, shares, contingencies and execution "
        "formalities carefully."
    ),
    "white-collar-defense-investigations": (
        "PRACTICE LENS — INVESTIGATIONS. Define scope, custodians, chronology, evidence "
        "provenance, legal theories, privilege, preservation and reporting audience. Separate "
        "established fact, conflicting evidence and inference; protect the integrity of the "
        "investigation record."
    ),
}


def _norm(value: str) -> str:
    import re

    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (value or "").lower())).strip("_")


def _lookup(table: dict[str, str], value: str) -> str:
    if value in table:
        return table[value]
    key = _norm(value)
    for name, text in table.items():
        if _norm(name) == key:
            return text
    return ""


def lab_prompts(*, practice_area: str = "", work_type: str = "") -> BenchPrompts:
    """LAB-wide profile plus optional practice-area and work-type overlays."""
    parts = [
        LAB_BASE.domain_guidance,
        _lookup(LAB_PRACTICE_AREAS, practice_area),
        _lookup(LAB_WORK_TYPES, work_type),
    ]
    guidance = "\n\n".join(p.strip() for p in parts if p and p.strip())
    return replace(LAB_BASE, domain_guidance=guidance)


CODEX_BASE_BODY = """\
You are an AI agent completing a professional work assignment inside a workspace.
The assignment message gives the absolute paths that apply to this run.

## Workspace layout

- `documents/` -- the assignment's source documents (the data room). Treat it as
  read-only: never modify, move, or delete anything under it.
- `output/` -- the deliverable directory. Every deliverable must be written to
  `output/<filename>` explicitly (or to its absolute path). Files left anywhere
  else are not part of the delivered work.
- `scripts/` -- helper scripts, run as `python3 scripts/<name>.py`.

## Reading documents

`.docx`, `.pdf`, `.pptx`, and `.xlsx` files are binary. To read one, run

    python3 scripts/read_doc.py <path>

which prints the extracted text. Plain text, markdown, and code files can be read
directly with shell tools such as `cat`.

## Doing the work

- Produce exactly the deliverable files the assignment names, with the exact
  filenames it asks for.
- Build binary deliverables (.docx/.xlsx/.pptx) programmatically with Python,
  then verify each produced file is non-empty and re-opens cleanly (for example
  by reading it back with `python3 scripts/read_doc.py`) before finishing.
- Ground the work in `documents/`: keep every figure, name, and claim traceable
  to a source document, and say so explicitly where the record is silent.
"""


_EXTRACT_SRC = r"""
import json, sys
from pathlib import Path

# Imported at driver top, not inside extract(): a reader that cannot be imported at all
# must exit non-zero here so deliverable_files raises, rather than degrading per file.
from evaluation.scoring import _read_file_as_text

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".xml", ".html", ".py",
                 ".yaml", ".yml", ".tsv"}
# The LAB judge's _read_file_as_text reads the WHOLE file, so a capped improver/selector
# view can be blinded to content the judge grades. CAP=0 means uncapped; the caller passes
# the cap (config.LAB_DELIV_CAP, fixed at 60_000).
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 0

def extract(p):
    try:
        return _read_file_as_text(p)
    except Exception:                          # per-FILE fallback only; a missing reader is
        if p.suffix.lower() in TEXT_SUFFIXES:  # a hard, visible failure at import above
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return f"(error reading {p.name}: {exc})"
        return f"(binary file: {p.name}, {p.stat().st_size} bytes)"

base = Path(sys.argv[1])
out = {}
if base.is_dir():
    for p in sorted(q for q in base.rglob("*") if q.is_file()):
        text = extract(p)
        if CAP and len(text) > CAP:
            text = text[:CAP] + f"\n\n... [truncated, {len(text) - CAP} more chars]"
        out[p.relative_to(base).as_posix()] = text
json.dump(out, sys.stdout)
"""

_BASE_BODY_SRC = r"""
import sys
from harness.run import DEFAULT_SKILLS
from opencode.render import compose_prompt, ENV_DELTA
body = compose_prompt(list(DEFAULT_SKILLS), skill_mode=sys.argv[1])
delta = ENV_DELTA.rstrip()
if delta and body.rstrip().endswith(delta):
    body = body.rstrip()[: -len(delta)].rstrip() + "\n"
sys.stdout.write(body)
"""

_RENDER_CHECK_SRC = r"""
import json, sys, tempfile
from pathlib import Path
try:
    from opencode.render import render_tree
    with tempfile.TemporaryDirectory() as td:
        rendered = render_tree(Path(td) / "ws", sys.argv[1], skill_mode=sys.argv[2])
    missing = [m for m in ("output/", "read_doc.py") if m not in rendered.prompt]
    print(json.dumps({"ok": not missing, "missing": missing, "error": ""}))
except Exception as exc:
    print(json.dumps({"ok": False, "missing": [],
                      "error": f"{type(exc).__name__}: {exc}"}))
"""


def _lab_env() -> dict[str, str]:
    return {
        **os.environ,
        # Children print document text; without UTF-8 mode a Windows console codec mangles it.
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(config.lab_repo()),
        # pandoc is on PATH for every LAB child: the judge and the deliverable extractor read
        # .docx through it, and a missing binary reads as "file not found", i.e. a silent 0.
        "PATH": os.pathsep.join(
            p
            for p in (
                str(config.LAB_PYTHON.parent),
                str(config.BUN.parent),
                f"{config.CODEX_DIST}/codex-path",
                _pandoc_dir(),
                os.environ.get("PATH", ""),
            )
            if p
        ),
    }


_READ_DOC = config.THIRD_PARTY / "lab_runner" / "opencode" / "scripts" / "read_doc.py"

_NUDGE = (
    "Continue the assignment to completion now. Do not ask for confirmation. "
    "If every deliverable has already been written to the output directory, "
    "reply with exactly: DONE"
)

# DSH mounts PowerShell, not bash, as the shell tool on Windows hosts.
DSH_SHELL_TOOL = "pwsh" if os.name == "nt" else "bash"

DSH_ENV_DELTA = """\
You are running under DeepSeek Harness. Three details differ from the description
above -- where they conflict, **these win**:

1. **Paths are real and absolute; the shell tool is `{shell}`.** It starts in the
   workspace root, whose absolute path is given in the assignment. `documents/`,
   `output/`, `scripts/` and `skills/` are subdirectories of it. There is no path
   rewriting and no `$WORKSPACE_DIR`-style variable; use the absolute paths. Where the
   description above says `bash`, use `{shell}`.

2. **Deliverables are not auto-routed.** Relative `write`/`edit` paths land
   wherever the tool resolves them, *not* in `output/`. Every deliverable must
   be written to `output/<filename>` explicitly (or to its absolute path).
   Anything left outside `output/` is not graded.

3. **`read` does not parse binary documents.** For `.docx`, `.pdf`, `.pptx`,
   and `.xlsx`, run:

       python scripts/read_doc.py <path>

   which prints the extracted text. `read` is still the right tool for plain
   text, markdown, and code. The file-type manuals (`docx`, `xlsx`, `pptx`) load
   through the `skill` tool.
"""


def _pandoc_dir() -> str:
    """The directory holding pandoc: PATH first, else the LAB venv's bundled binary."""
    found = shutil.which("pandoc")
    if found:
        return str(Path(found).parent)
    site = config.LAB_PYTHON.parent.parent / ("Lib" if os.name == "nt" else "lib")
    for p in site.rglob("pypandoc/files/pandoc*"):
        if p.is_file():
            return str(p.parent)
    return ""


def _judge_env(judge_model: str) -> dict[str, str]:
    """The judge's environment. A `compat/<model>` judge reads its endpoint from
    HARNESS2_JUDGE_BASE_URL (default: OpenCode Go) and its key from HARNESS2_JUDGE_API_KEY,
    else from the DSH credential store; neither is written anywhere."""
    env = _lab_env()
    if not judge_model.startswith("compat/"):
        return env
    base = os.environ.get("HARNESS2_JUDGE_BASE_URL") or "https://opencode.ai/zen/go/v1"
    key = os.environ.get("HARNESS2_JUDGE_API_KEY") or DSH.credential("OPENCODE_GO_API_KEY")
    if not key:
        raise RuntimeError(
            "compat judge has no key: set HARNESS2_JUDGE_API_KEY or store OPENCODE_GO_API_KEY "
            "in DSH (the Models page)"
        )
    headers = json.loads(os.environ.get("HARNESS2_JUDGE_HEADERS") or "{}")
    if "opencode.ai" in base:
        # OpenCode Go routes by session and rejects requests without one.
        headers.setdefault("x-opencode-session", str(uuid.uuid4()))
    env.update(
        HARNESS2_JUDGE_BASE_URL=base,
        HARNESS2_JUDGE_API_KEY=key,
        HARNESS2_JUDGE_HEADERS=json.dumps(headers),
    )
    return env


def _dsh_lab_env() -> dict[str, str]:
    """A rollout's environment: `python` is the LAB interpreter (its document parsers and
    the skill helpers' dependencies), documents parse on the host, output is UTF-8."""
    env = _lab_env()
    env["LAB_DOC_PARSER"] = "host"
    env["LAB_BENCH_ROOT"] = str(config.lab_repo())
    return env


def _json_or_none(p: Path) -> dict | None:
    """Parse `p` as JSON; None when it is missing or torn."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_detached(cmd: list[str], *, timeout: int, **kw) -> None:
    """Run `cmd` in its own process group and kill the group on a wall timeout."""
    proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            kill_group(proc, SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark runner exited with status {proc.returncode}")


def _require_runner(path: Path) -> None:
    if not Path(path).is_file():
        raise RuntimeError(
            f"upstream LAB runner missing: {path}. The upstream clone does not track it at "
            f"the pinned commit; it ships as the tracked overlay third_party/lab_runner/ — "
            f"run scripts/setup_benchmark.py <benchmark> to copy it into place."
        )


@cache
def _opencode_base_body(skill_mode: str) -> str:
    """Composed once per process per mode: composing reads every SKILL.md."""
    _require_runner(config.lab_repo() / "opencode" / "render.py")
    try:
        r = subprocess.run(
            [str(config.LAB_PYTHON), "-c", _BASE_BODY_SRC, skill_mode],
            cwd=config.lab_repo(),
            env=_lab_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"base-body compose failed to run: {exc}") from exc
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(
            f"base-body compose failed under {config.LAB_PYTHON} "
            f"(rc={r.returncode}): {r.stderr.strip()[-800:]}"
        )
    return r.stdout


def _runtime_scripts() -> frozenset[str]:
    """Script basenames the LAB opencode runtime puts into every workspace, so a harness may
    not ship them. Derived from the upstream skills directory because a hand-listed set
    goes stale, and the staleness shows up only as fallbacks. An unmaterialised checkout
    yields the empty set, which `_require_runtime_scripts` refuses at preflight."""
    skills = config.lab_repo() / "harness" / "skills"
    return frozenset(p.name for p in skills.glob("*/scripts/*.py"))


def _require_runtime_scripts() -> frozenset[str]:
    """`_runtime_scripts()`, refusing the empty set. Called from preflight, never from the spec."""
    names = _runtime_scripts()
    if not names:
        skills = config.lab_repo() / "harness" / "skills"
        raise FileNotFoundError(
            f"no skill scripts under {skills} — an empty exemption set is precisely the "
            f"condition this derivation exists to prevent, and it fails silently as mass "
            f"fell_back rather than as an error; materialise the LAB checkout with "
            f"scripts/setup_benchmark.py lab"
        )
    return names


def lab_spec(substrate_name: str) -> BenchSpec:
    """The LAB BenchSpec for one substrate cell. A value, not a code path."""
    if substrate_name not in ("oc", "cx", "dsh"):
        raise ValueError(f"substrate must be 'oc', 'cx' or 'dsh', got {substrate_name!r}")
    if substrate_name == "dsh":
        channel = resolve_channel_model("dsh", config.DSH_IMPROVER_MODEL, effort="high")
    else:
        channel = resolve_channel_model(None, None, effort="high")
    install_mode = {"oc": "workspace_config", "cx": "codex_root", "dsh": DSH.INSTALL_MODE}
    reserved = frozenset()
    if substrate_name == "oc":
        reserved = _runtime_scripts()
    elif substrate_name == "dsh":
        reserved = _runtime_scripts() | {"read_doc.py"}
    return BenchSpec(
        name="lab",
        runs_root=config.RUNS / "lab",
        install_mode=install_mode[substrate_name],
        prompts=LAB_BASE,
        channels={"proposer": channel, "selector": channel},
        domain_word="practice area",
        leak_globs=("task.json",),
        allow_subagents=False,
        reserved_scripts=reserved,
        proposer_deliverables="contents",
        obs_trunc=50_000,
    )


class LabAdapter:
    """The BenchmarkAdapter surface for LAB, on either substrate.

    skill_mode: `native` loads the base skill manuals on demand; `inline` puts the
    docx/xlsx/pptx manuals in the always-on body, which every proposal must re-emit.
    Switching moves the recorded baseline.
    """

    def __init__(
        self,
        *,
        substrate: str = "oc",
        model: str | None = None,
        timeout: int | None = None,
        skill_mode: str = "native",
        solver_effort: str = "high",
        judge_model: str = JUDGE_MODEL,
        split_file: Path | None = None,
    ):
        self.split_file = split_file
        if skill_mode not in ("native", "inline"):
            raise ValueError(f"skill_mode must be 'native' or 'inline', got {skill_mode!r}")
        self.spec = lab_spec(substrate)
        if substrate == "oc":
            self.substrate = OpencodeSubstrate(launcher=self._launch_oc)
            self.model = model or config.SOLVER_MODEL_OC
            self._results_root = config.RUNS / "lab" / "results" / "oc"
        elif substrate == "dsh":
            self.substrate = DshSubstrate(launcher=self._launch_dsh)
            self.model = model or DSH.SOLVER_MODEL
            self._results_root = config.RUNS / "lab" / "results" / "dsh"
        else:
            self.substrate = CodexSubstrate(launcher=self._launch_cx)
            self.model = model or config.SOLVER_MODEL_CX

            self._results_root = config.RUNS / "lab" / "results" / "cx"
        self.timeout = int(timeout or 3600)
        self.skill_mode = skill_mode
        self.solver_effort = solver_effort
        self.judge_model = judge_model
        self._deliverables: dict[str, dict[str, str]] = {}

    def tasks(self, split: str = "all") -> list[str]:
        return select(discover("lab"), split, self.split_file, benchmark="lab")

    def domain_of(self, task_id: str) -> str:
        return task_id.split("/", 1)[0]

    def reference_domain_order(self) -> list[str]:
        return sorted(set(discover("lab").values()))

    def prompts_for(self, task_id: str) -> BenchPrompts:
        """Practice-area + work-type overlay; criteria stripped at the same read boundary."""
        task = self._task_json(task_id)
        return lab_prompts(
            practice_area=self.domain_of(task_id), work_type=str(task.get("work_type", ""))
        )

    def _task_json(self, task_id: str) -> dict:
        """task.json WITHOUT `criteria` — the ONLY place this adapter reads a task."""
        p = config.lab_repo() / "tasks" / task_id / "task.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        data.pop("criteria", None)
        return data

    def task_prompt(self, task_id: str, *, max_chars: int = 12_000) -> str:
        """Answer-free markdown rendering of a task (the frozen tts_lab wording)."""
        t = self._task_json(task_id)
        parts = [
            f"# TASK {task_id}",
            f"**Title:** {t.get('title', '')}",
            f"**Work type:** {t.get('work_type', '')}",
        ]
        tags = t.get("tags") or []
        if tags:
            parts.append("**Tags:** " + ", ".join(str(x) for x in tags))
        parts.append("\n## Instructions\n" + str(t.get("instructions", "")).strip())
        deliv = t.get("deliverables") or {}
        if deliv:
            lines = [
                f"- `{name}`" + (f" — {desc}" if desc and desc != name else "")
                for name, desc in deliv.items()
            ]
            parts.append("\n## Deliverables\n" + "\n".join(lines))
        return "\n\n".join(parts)[:max_chars]

    def task_files(self, task_id: str) -> Path | None:
        """The data room the work was done against. `criteria` lives in task.json, not here."""
        d = config.lab_repo() / "tasks" / task_id / "documents"
        return d if d.is_dir() else None

    def _run_dir(self, run_id: str) -> Path:
        return self._results_root / run_id

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        out = self._run_dir(run_id)
        if RM.is_complete(out, harness_dir=harness_dir) and self.artifacts_intact(out):
            self._check_tool_gate(out)
            return out
        if out.is_dir() and any(out.iterdir()):
            reason = RM.why_not(out, harness_dir=harness_dir)
            if reason == "reusable":
                reason = "output fingerprint missing or changed"
            print(
                f"  [lab-{self.substrate.name}] re-rolling {run_id}: {reason}",
                flush=True,
            )
            self._safe_rmtree(out)
        self._deliverables.pop(str(out.resolve()), None)
        install = self._install(harness_dir)
        try:
            self.substrate.execute(task_id, install, [run_id], int(timeout or self.timeout))
        finally:
            if install.extras.get("harness") == "staged":
                shutil.rmtree(install.staging_dir, ignore_errors=True)
        if not (out / "metrics.json").is_file():
            raise RuntimeError(
                f"LAB rollout did not produce metrics.json: {out} — the runner refuses to "
                f"write it when the session stream had no parseable events (dead chain / "
                f"missing binary), so this failure must not be cached"
            )
        self._check_tool_gate(out)
        RM.mark_complete(
            out,
            run_id=run_id,
            harness_dir=harness_dir,
            extra={"output_fingerprint": self._output_fingerprint(out)},
        )
        return out

    def _install(self, harness_dir: Path | None) -> Install:
        """BASE (harness_dir=None) is a value: fingerprint 'base', and the launcher decides
        what 'base' means on its substrate."""
        if harness_dir is None:
            return Install(
                staging_dir=self.base_harness(), fingerprint="base", extras={"harness": "base"}
            )
        inst = self.substrate.materialise(Path(harness_dir), self.spec, {})
        return Install(
            staging_dir=inst.staging_dir,
            fingerprint=inst.fingerprint,
            extras={**inst.extras, "harness": "staged"},
        )

    def _launch_oc(self, req: RunRequest, install: Install) -> list[Path]:
        runner = config.lab_repo() / "opencode" / "run_opencode.py"
        _require_runner(runner)
        dirs = []
        for run_id in req.run_ids:
            cmd = [
                str(config.LAB_PYTHON),
                "-u",
                str(runner),
                "--task",
                req.task_ref,
                "--run-id",
                run_id,
                "--results-root",
                str(self._results_root),
                "--model",
                self.model,
                "--tool-mode",
                "base",
                "--reasoning-effort",
                self.solver_effort,
                "--skill-mode",
                self.skill_mode,
                "--timeout",
                str(req.timeout),
                "--opencode-dir",
                str(config.OPENCODE_DIR),
                "--bun",
                str(config.BUN),
            ]
            if install.extras.get("harness") != "base":
                cmd += ["--harness-dir", str(install.staging_dir)]
            try:
                run_detached(
                    cmd,
                    cwd=config.lab_repo(),
                    env=_lab_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=req.timeout * 2 + 600,
                )
            except subprocess.TimeoutExpired:
                print(f"  [lab-oc] runner wall-timeout for {req.task_ref} @ {run_id}", flush=True)
            dirs.append(self._run_dir(run_id))
        return dirs

    def _launch_cx(self, req: RunRequest, install: Install) -> list[Path]:
        runner = config.lab_repo() / "external" / "adapt_runs_cx" / "run_codex_lab.py"
        _require_runner(runner)
        dirs = []
        for run_id in req.run_ids:
            cmd = [
                str(config.LAB_PYTHON),
                "-u",
                str(runner),
                "--task",
                req.task_ref,
                "--run-id",
                run_id,
                "--model",
                self.model,
                "--reasoning-effort",
                self.solver_effort,
                "--timeout",
                str(req.timeout),
                "--harness-dir",
                str(install.staging_dir),
            ]

            env = {
                **_lab_env(),
                "ADAPT_LABCX_CODEX_BIN": str(Path(config.CODEX_DIST) / "codex"),
                "ADAPT_LABCX_RESULTS": str(self._results_root),
                "ADAPT_LABCX_SCRATCH": os.environ.get("ADAPT_LABCX_SCRATCH")
                or str(config.RUNS / "lab" / "scratch"),
                "ADAPT_LABCX_BASE_URL": os.environ.get("ADAPT_LABCX_BASE_URL")
                or config.CODEX_BASE_URL,
            }
            if config.VERTEX_PROXY_KEY:
                env["VERTEX_PROXY_KEY"] = config.VERTEX_PROXY_KEY

            try:
                run_detached(
                    cmd,
                    cwd=config.lab_repo(),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=req.timeout + 900,
                )
            except subprocess.TimeoutExpired:
                print(f"  [lab-cx] runner wall-timeout for {req.task_ref} @ {run_id}", flush=True)
            dirs.append(self._run_dir(run_id))
        return dirs

    def _launch_dsh(self, req: RunRequest, install: Install) -> list[Path]:
        """One DSH rollout per run id, laid out like `run_opencode.py` so `run_eval` scores it
        unchanged: results/<run_id>/{workspace,output,metrics.json,config.json}."""
        task = self._task_json(req.task_ref)
        dirs = []
        for run_id in req.run_ids:
            results_dir = self._run_dir(run_id)
            dirs.append(results_dir)
            staged = install
            if install.extras.get("harness") == "base":
                # Base is staged like any harness so runtime files never touch the shared tree.
                fresh = self.substrate.materialise(self.base_harness(), self.spec, {})
                staged = replace(fresh, extras={**fresh.extras, "harness": "base"})
            try:
                self._run_dsh(req, task, run_id, results_dir, staged)
            finally:
                if staged is not install:
                    shutil.rmtree(staged.staging_dir, ignore_errors=True)
        return dirs

    def _run_dsh(
        self, req: RunRequest, task: dict, run_id: str, results_dir: Path, staged: Install
    ) -> None:
        workspace = results_dir / "workspace"
        output_dir = workspace / "output"
        docs = workspace / "documents"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(config.lab_repo() / "tasks" / req.task_ref / "documents", docs)
        for p in docs.rglob("*"):
            if p.is_file():
                p.chmod(0o444)
        (workspace / "scripts").mkdir(exist_ok=True)
        shutil.copy2(_READ_DOC, workspace / "scripts" / "read_doc.py")
        # The base skill manuals are runtime, not harness: catalogued from the staged skills
        # root, their helpers at the workspace path the manuals name.
        lab_skills = config.lab_repo() / "harness" / "skills"
        for skill in sorted(p.parent for p in lab_skills.glob("*/SKILL.md")):
            shutil.copytree(
                skill, Path(staged.staging_dir) / "skills" / skill.name, dirs_exist_ok=True
            )
            if (skill / "scripts").is_dir():
                shutil.copytree(
                    skill / "scripts",
                    workspace / "skills" / skill.name / "scripts",
                    dirs_exist_ok=True,
                )
        DSH.install_scripts(staged, workspace)
        agents = Path(staged.staging_dir) / "AGENTS.md"
        agents.write_text(
            agents.read_text(encoding="utf-8").rstrip()
            + "\n\n---\n\n"
            + DSH_ENV_DELTA.replace("{shell}", DSH_SHELL_TOOL),
            encoding="utf-8",
        )
        env = self.substrate.run_env(_dsh_lab_env(), staged, self.model)
        prompt = (
            f"=== WORKSPACE ===\n{workspace}\n\n"
            f"=== DOCUMENTS (read-only) ===\n{docs}\n\n"
            f"=== OUTPUT DIRECTORY (deliverables go here) ===\n{output_dir}\n\n"
            f"=== ASSIGNMENT ===\n{task['instructions']}"
        )
        stream = results_dir / DSH.STREAM_NAME
        rc, wall, tail = DSH.run_headless(prompt, workspace, env, stream, timeout=req.timeout)
        nudges = 0
        while nudges < 2 and rc == 0 and not any(p.is_file() for p in output_dir.rglob("*")):
            sid = DSH.session_id(stream)
            remaining = req.timeout - wall
            if not sid or remaining < 60:
                break
            nudges += 1
            more = stream.with_name(f"dsh_session.nudge{nudges}.jsonl")
            rc, w, tail = DSH.run_headless(
                _NUDGE, workspace, env, more, timeout=int(remaining), session_id=sid
            )
            wall += w
            with stream.open("ab") as fh:
                fh.write(more.read_bytes())
            more.unlink()
        graded = results_dir / "output"
        graded.mkdir(parents=True, exist_ok=True)
        for src in output_dir.rglob("*"):
            if src.is_file():
                dest = graded / src.relative_to(output_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        usage = DSH.token_usage(stream)
        config_json = {
            "runner": "dsh",
            "model": self.model,
            "task": req.task_ref,
            "run_id": run_id,
            "profile": DSH.PROFILE,
            "harness": staged.extras.get("harness", "staged"),
            "nudges": nudges,
        }
        (results_dir / "config.json").write_text(
            json.dumps(config_json, indent=2), encoding="utf-8"
        )
        if usage["steps"] == 0 and not DSH.final_text(stream):
            # Mirrors run_opencode.py: no parseable events means a dead chain, which must
            # not be cached as a rollout. execute() raises on the missing metrics.json.
            (results_dir / "dsh_stderr_tail.txt").write_text(tail, encoding="utf-8")
            return
        metrics = {
            "runner": "dsh",
            "model": self.model,
            "task": req.task_ref,
            "run_id": run_id,
            "turn_count": usage["steps"],
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "cache_read_tokens": usage["cache_read"],
            "total_tokens": usage["input"] + usage["output"],
            "wall_clock_seconds": wall,
            "cost_usd": 0.0,
            "finished_cleanly": rc == 0 and DSH.completed(stream),
            "exit_code": rc,
            "tool_calls": sum(1 for e in DSH.iter_events(stream) if e.get("type") == "tool_call"),
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _safe_rmtree(self, d: Path) -> None:
        """rm -rf restricted to this cell's results root — a guard, not a convenience."""
        d = Path(d).resolve()
        root = self._results_root.resolve()
        if root != d and root not in d.parents:
            raise RuntimeError(f"refusing to remove {d}: outside the LAB results root {root}")
        shutil.rmtree(d, ignore_errors=True)

    def _output_fingerprint(self, run_dir: Path) -> str | None:
        return RM.fingerprint(Path(run_dir) / "output") if self.delivered(run_dir) else None

    def artifacts_intact(self, run_dir: Path) -> bool:
        marker = RM.read_mark(run_dir) or {}
        return (
            (run_dir / "metrics.json").is_file()
            and "output_fingerprint" in marker
            and marker["output_fingerprint"] == self._output_fingerprint(run_dir)
        )

    def delivered(self, run_dir: Path) -> bool:
        output_dir = Path(run_dir) / "output"
        return any(path.is_file() for path in output_dir.rglob("*"))

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        """The substrate's ONE renderer over the raw session stream, effectively uncapped."""
        name = {"oc": "opencode_session.jsonl", "dsh": DSH.STREAM_NAME}.get(
            self.substrate.name, "codex_session.jsonl"
        )
        stream = Path(run_dir) / name
        if not stream.is_file():
            return f"(no trajectory recorded for {Path(run_dir).name})"
        return self.substrate.read_trajectory(stream, obs_trunc=obs_trunc, max_chars=10**9)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        """filename -> extracted text via the JUDGE's own reader, in a subprocess inside
        the upstream repo (cached per run dir). An extraction failure RAISES."""
        key = str(Path(run_dir).resolve())
        if key in self._deliverables:
            return dict(self._deliverables[key])
        try:
            r = subprocess.run(
                [
                    str(config.LAB_PYTHON),
                    "-c",
                    _EXTRACT_SRC,
                    str(Path(run_dir) / "output"),
                    str(config.LAB_DELIV_CAP),
                ],
                cwd=config.lab_repo(),
                env=_lab_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"deliverable extraction failed for {run_dir}: {exc}") from exc
        if r.returncode != 0:
            raise RuntimeError(
                f"deliverable extraction failed for {run_dir} "
                f"(rc={r.returncode}): {r.stderr.strip()[-500:]}"
            )
        files = json.loads(r.stdout)
        self._deliverables[key] = files
        return dict(files)

    def deliverable_text(self, run_dir: Path) -> str:
        """The CONTENTS, not the file names."""
        files = self.deliverable_files(run_dir)
        if not files:
            return "(this run produced NO deliverables)"
        return "\n\n".join(
            f"----- FILE: {name} -----\n{text.strip() or '(empty file)'}"
            for name, text in files.items()
        )

    def deliverable_images(self, run_dir: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        for sub in ("output", "workspace/output"):
            d = Path(run_dir) / sub
            if d.is_dir():
                return [
                    p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in exts
                ][:8]
        return []

    def base_harness(self) -> Path:
        """The unmodified tree this cell starts from, materialised once per process and per
        pid: the domain threads of one sweep share the path."""
        if self.substrate.name == "oc":
            d = self.spec.runs_root / f"_base_harness_{self.skill_mode}_{os.getpid()}"
        elif self.substrate.name == "dsh":
            d = self.spec.runs_root / f"_base_harness_dsh_{os.getpid()}"
        else:
            d = self.spec.runs_root / f"_base_harness_cx_{os.getpid()}"
        with _BASE_LOCK:
            if d not in _BASE_DONE:
                if self.substrate.name == "oc":
                    HT.write_tree(
                        OPENCODE_SURFACE,
                        {
                            "systemprompt.md": _opencode_base_body(self.skill_mode),
                            "guardrails.md": "",
                            "LongTermMEMORY.md": "",
                            "tool_descriptions/tool_guidance.md": "",
                        },
                        d,
                    )
                elif self.substrate.name == "dsh":
                    HT.write_tree(
                        DSH.DSH_SURFACE,
                        {
                            # The same native-skill body the oc cell starts from; the DSH
                            # environment delta is runtime, appended at launch.
                            "systemprompt.md": _opencode_base_body("native"),
                            "guardrails.md": "",
                            "LongTermMEMORY.md": "",
                            "tool_descriptions/tool_guidance.md": "",
                        },
                        d,
                    )
                else:
                    HT.write_tree(CODEX_SURFACE, {"AGENTS.md": CODEX_BASE_BODY}, d)
                _BASE_DONE.add(d)
        return d

    def validate_tree(self, tree: dict[str, str]) -> None:
        """LAB environment invariants: deliverables go to output/, binary documents are read
        via scripts/read_doc.py. oc checks the RENDERED prompt, cx the AGENTS.md body."""
        if self.substrate.name == "oc":
            self._validate_oc(tree)
        elif self.substrate.name == "dsh":
            self._validate_dsh(tree)
        else:
            self._validate_cx(tree)

    def _validate_dsh(self, tree: dict[str, str]) -> None:
        """The output/ and read_doc.py rules live in the runtime delta, so only the surface
        rules and the runtime-owned skill names can be broken here."""
        errs = DSH.dsh_tree_problems(tree)
        lab_skills = {
            p.parent.name for p in (config.lab_repo() / "harness" / "skills").glob("*/SKILL.md")
        }
        for name in sorted(set(HT.skill_names(DSH.DSH_SURFACE, tree)) & lab_skills):
            errs.append(
                f"skills/{name}/SKILL.md: `{name}` is a runtime-provided skill manual in this "
                f"environment; give a new skill a different name instead of replacing it"
            )
        if errs:
            raise ValueError(" | ".join(errs))

    def _validate_oc(self, tree: dict[str, str]) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = HT.write_tree(OPENCODE_SURFACE, tree, Path(td) / "h", stubs=False)
            try:
                r = subprocess.run(
                    [str(config.LAB_PYTHON), "-c", _RENDER_CHECK_SRC, str(d), self.skill_mode],
                    cwd=config.lab_repo(),
                    env=_lab_env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=300,
                )
                got = json.loads(r.stdout)
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                raise ValueError(f"harness does not render: {exc}") from exc
        if got.get("error"):
            raise ValueError(f"harness does not render: {got['error']}")
        errs = []
        if "output/" in got.get("missing", []):
            errs.append(
                "the rendered prompt must say deliverables go to `output/<filename>` "
                "explicitly -- relative write/edit paths are NOT auto-routed and "
                "anything outside output/ is invisible to grading"
            )
        if "read_doc.py" in got.get("missing", []):
            errs.append(
                "the rendered prompt must say .docx/.pdf/.pptx/.xlsx are readable only "
                "via `python3 scripts/read_doc.py <path>` -- the `read` tool cannot "
                "parse them and the agent will silently 'read' binary noise"
            )
        if errs:
            raise ValueError(" | ".join(errs))

    def _validate_cx(self, tree: dict[str, str]) -> None:
        body = HT.always_on_text(CODEX_SURFACE, tree)
        errs = []
        if "output/" not in body:
            errs.append(
                "AGENTS.md must say deliverables go to `output/<filename>` "
                "explicitly -- under codex nothing routes relative paths, and "
                "anything left outside output/ is invisible to grading. Add the "
                "output/ rule back to AGENTS.md."
            )
        if "read_doc.py" not in body:
            errs.append(
                "AGENTS.md must say .docx/.pdf/.pptx/.xlsx are readable only via "
                "`python3 scripts/read_doc.py <path>` -- shell reads of binary "
                "documents return noise, silently. Add the read_doc.py rule back "
                "to AGENTS.md."
            )
        if errs:
            raise ValueError(" | ".join(errs))

    def _tool_gate_problems(self, run_dir: Path) -> list[str]:
        """Disagreement between the emitted permission and tools allowlists; no config file
        is not a conflict."""
        cfg = Path(run_dir) / "workspace" / "opencode.json"
        if not cfg.is_file():
            return []
        c = _json_or_none(cfg)
        if c is None:
            return []
        perm = c.get("permission") or {}
        tools = c.get("tools") or {}
        problems = []
        for name, enabled in tools.items():
            want = "allow" if enabled else "deny"
            if perm.get(name) != want:
                problems.append(
                    f"tool gate conflict for {name!r}: tools={enabled!r}, "
                    f"permission={perm.get(name)!r}; expected {want!r}"
                )
        return problems

    def _check_tool_gate(self, run_dir: Path) -> None:
        if problems := self._tool_gate_problems(run_dir):
            raise RuntimeError(" | ".join(problems))

    def preflight(self) -> None:
        """Everything money-adjacent, checked before a rollout is paid for."""
        self.substrate.preflight()
        if not config.LAB_PYTHON.is_file():
            raise RuntimeError(
                f"LAB interpreter missing at {config.LAB_PYTHON}; run python scripts/setup_benchmark.py lab"
            )
        if self.substrate.name == "oc":
            _require_runner(config.lab_repo() / "opencode" / "run_opencode.py")
            _require_runtime_scripts()
        elif self.substrate.name == "dsh":
            _require_runner(config.lab_repo() / "opencode" / "render.py")
            _require_runtime_scripts()
            if not _pandoc_dir():
                raise RuntimeError(
                    "pandoc not found (the docx reader needs it): install pandoc, or "
                    f"`uv pip install --python {config.LAB_PYTHON} pypandoc_binary`"
                )
        else:
            _require_runner(config.lab_repo() / "external" / "adapt_runs_cx" / "run_codex_lab.py")
        try:
            r = subprocess.run(
                [str(config.LAB_PYTHON), "-c", "import evaluation.scoring"],
                cwd=config.lab_repo(),
                env=_lab_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"could not probe the LAB judge reader: {exc}") from exc
        if r.returncode != 0:
            raise RuntimeError(
                f"`import evaluation.scoring` fails under {config.LAB_PYTHON} "
                f"({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'}) "
                f"— run python scripts/setup_benchmark.py lab to install its dependency environment"
            )

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """LAB's own judge (evaluation.run_eval). Judge failure => None, NEVER 0, and
        nothing is cached. The ABSOLUTE run dir is passed: run_eval joins its own
        RESULTS_DIR with it, and an absolute right-hand side wins that join."""
        p = Path(run_dir) / "scores.json"
        if not p.is_file():
            cmd = [
                str(config.LAB_PYTHON),
                "-u",
                "-m",
                "evaluation.run_eval",
                "--run-id",
                str(Path(run_dir).resolve()),
                "--task",
                task_id,
                "--judge-model",
                self.judge_model,
            ]
            log = Path(run_dir) / "judge.log"
            try:
                with log.open("w", encoding="utf-8") as out:
                    run_detached(
                        cmd,
                        cwd=config.lab_repo(),
                        env=_judge_env(self.judge_model),
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                    )
            except (OSError, subprocess.TimeoutExpired):
                return None
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}; see {log}") from None
        s = _json_or_none(p)
        if s is None:
            return None
        return {
            "n_passed": s.get("n_passed"),
            "n_criteria": s.get("n_criteria"),
            "all_pass": s.get("all_pass"),
        }


__all__ = ["LabAdapter", "lab_spec", "lab_prompts", "LAB_BASE", "CODEX_BASE_BODY", "JUDGE_MODEL"]

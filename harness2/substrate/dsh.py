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

"""DshSubstrate: DeepSeek Harness (DSH) as a solver substrate.

One install mode, `dsh_agents_home`: the editable tree is staged into a directory that the
rollout's DSH process adopts as its shared agents home (`DSH_AGENTS_HOME`). DSH's
agent-instructions plugin loads `<agents home>/AGENTS.md` as the user-global instruction
chain, so the composed always-on text arrives there; the profile's skill-filesystem plugin
scans only `H2_SKILLS_DIR`, so the staged skills are the whole skill catalog.

The editable surface keeps OpenCode's always-on file names on purpose: an improver session
also runs under DSH, and DSH loads any `AGENTS.md` it reads its way into, so a harness file
literally named `AGENTS.md` would be applied to the improver editing it. The name exists only
in the staged copy.

DSH has no tool-loop middleware a harness can ship (its plugins are host packages, not
per-run files), so c8 is absent from this surface rather than read as prose.

The runner boots the `h2` profile (see docs/DSH.md) with `dsh --profile h2 --json -`,
the task on stdin and the rollout workspace as cwd; the sandbox's workspace root is that cwd.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping

from .. import _compat
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest

DSH_SURFACE = SurfaceSpec(
    name="dsh",
    always_on=(
        "systemprompt.md",
        "guardrails.md",
        "LongTermMEMORY.md",
        "tool_descriptions/tool_guidance.md",
    ),
    always_on_max=24_000,
    always_on_cap_reason="every token here competes with the agent's own output budget",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        "Fix it by adding a trigger line naming `{name}` to systemprompt.md or "
        'guardrails.md, e.g. "When <the situation this skill handles>, load the '
        '`{name}` skill."'
    ),
    autotrigger_file="guardrails.md",
    trigger_line="- When {desc}, load the `{name}` skill.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated subagents, run through the `subagent` tool in their OWN "
            "context. Add one as sub_agents/<name>/AGENT.md with a `description:` "
            "frontmatter line. -->\n",
        ),
        (
            "skills/README.md",
            "<!-- c6: reusable workflows loaded on demand via the `skill` tool. Add one as "
            "skills/<name>/SKILL.md with frontmatter `name:` matching the directory (kebab-case) "
            "and a `description:`. -->\n",
        ),
    ),
)

INSTALL_MODE = "dsh_agents_home"
STREAM_NAME = "dsh_session.jsonl"

PROFILE = os.environ.get("HARNESS2_DSH_PROFILE", "h2")
SOLVER_MODEL = os.environ.get("HARNESS2_DSH_SOLVER_MODEL", "opencode-go/deepseek-v4-pro")

# DSH's skill-filesystem drops a skill whose frontmatter name is not kebab-case, with no
# diagnostic the model can see; the shared tree grammar also admits `_`.
_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def dsh_tree_problems(tree: Mapping[str, str]) -> list[str]:
    """Surface rules the shared validator cannot know: DSH's kebab-case skill names."""
    problems = []
    for name in sorted(HT.skill_names(DSH_SURFACE, dict(tree))):
        if not _KEBAB.match(name):
            problems.append(
                f"skills/{name}/SKILL.md: DSH loads only kebab-case skill names "
                f"(lowercase words joined by '-'); rename the directory and its `name:` line"
            )
    return problems


def _frontmatter_desc(body: str) -> str:
    m = re.search(r"^description:\s*(.+)$", body[:600], re.M)
    return m.group(1).strip() if m else "(no description)"


def compose_agents_md(staged: Path) -> str:
    """The always-on text DSH injects for one rollout.

    Skills need no index here: DSH advertises the staged catalog itself. Subagents are not
    DSH presets, so each is advertised with the instruction to pass its role verbatim to the
    generic `subagent` tool. Scripts are named where `install_scripts` puts them: the
    workspace's own `scripts/`, so the `python scripts/<name>.py` an editor writes resolves.
    """
    parts = [
        t
        for f in DSH_SURFACE.always_on
        if (staged / f).is_file() and (t := (staged / f).read_text(errors="replace").strip())
    ]
    agents = sorted(staged.glob("sub_agents/*/AGENT.md"), key=lambda p: p.parent.name)
    if agents:
        lines = "\n".join(
            f"- `{p.parent.name}` ({_frontmatter_desc(p.read_text(errors='replace'))}): "
            f"role file `{p.as_posix()}`"
            for p in agents
        )
        parts.append(
            "# AVAILABLE SUBAGENTS\nDelegate with the `subagent` tool when the situation "
            "matches. The child starts with an empty conversation: put the role file's full "
            f"text, then the bounded sub-task, in the delegated prompt.\n{lines}"
        )
    scripts = sorted(p.name for p in staged.glob("scripts/*.py"))
    if scripts:
        parts.append(
            f"# HELPER SCRIPTS\nAvailable in the workspace at `scripts/`: {', '.join(scripts)}\n"
            "Run with `python scripts/<name>.py`."
        )
    return "\n\n---\n\n".join(parts) + "\n"


def install_scripts(install: Install, workspace: Path) -> list[str]:
    """Copy the staged c7 helpers into `<workspace>/scripts/`; return the names copied."""
    names = []
    for p in sorted(Path(install.staging_dir).glob("scripts/*.py")):
        dest = Path(workspace) / "scripts" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        names.append(p.name)
    return names


def dsh_command() -> list[str]:
    """argv prefix that starts DSH; `HARNESS2_DSH_BIN` overrides the PATH lookup."""
    exe = os.environ.get("HARNESS2_DSH_BIN") or shutil.which("dsh")
    if not exe:
        raise RuntimeError(
            "dsh not found: install DeepSeek Harness (npm i -g @deepseek-ai/dsh) or set "
            "HARNESS2_DSH_BIN"
        )
    return [exe]


def split_model(model: str) -> tuple[str, str]:
    provider, sep, name = model.partition("/")
    if not sep or not provider or not name:
        raise ValueError(f"DSH model ids are provider/model, got {model!r}")
    return provider, name


def dsh_env(
    base_env: Mapping[str, str], *, agents_home: Path, skills_dir: Path | None, model: str
) -> dict[str, str]:
    """The per-run variables the `h2` profile reads; see the profile's patch layer."""
    provider, name = split_model(model)
    env = dict(base_env)
    env["DSH_AGENTS_HOME"] = str(agents_home)
    if skills_dir is not None:
        env["H2_SKILLS_DIR"] = str(skills_dir)
    else:
        env.pop("H2_SKILLS_DIR", None)
    env["H2_PROVIDER"] = provider
    env["H2_MODEL"] = name
    env.setdefault("DSH_PERMISSION_MODE", "workspace-write")
    env.setdefault("HOME", str(Path.home()))
    return env


def run_headless(
    prompt: str,
    cwd: Path,
    env: Mapping[str, str],
    stream_path: Path,
    *,
    timeout: int,
    profile: str = PROFILE,
) -> tuple[int, float, str]:
    """One headless DSH turn: events to `stream_path`; returns (rc, wall seconds, stderr tail).

    On a wall timeout the whole process tree is killed (the launcher is a shim over node) and
    the rc is -1; whatever the stream holds by then is the rollout's record.
    """
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = stream_path.with_suffix(".stderr.log")
    cmd = dsh_command() + ["--profile", profile, "--json", "-"]
    t0 = time.time()
    with open(stream_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            start_new_session=not _compat.IS_WINDOWS,
        )
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _compat.kill_group(proc, _compat.SIGKILL)
            proc.wait()
            rc = -1
    tail = err_path.read_text(encoding="utf-8", errors="replace")[-600:].strip()
    return rc, round(time.time() - t0, 1), tail


class DshSubstrate:
    """DSH behind the one Substrate interface; the launcher body is benchmark-supplied."""

    name = "dsh"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return DSH_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        if spec.install_mode != INSTALL_MODE:
            raise ValueError(
                f"install_mode {spec.install_mode!r} is not a DSH mode; this substrate "
                f"supports {INSTALL_MODE!r}"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(DSH_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a DSH harness tree (no always-on file present)")
        problems = dsh_tree_problems(HT.read_tree(DSH_SURFACE, tree_dir))
        if problems:
            raise ValueError("; ".join(problems))
        staged = Path(tempfile.mkdtemp(prefix="adapt-dsh-"))
        try:
            shutil.copytree(tree_dir, staged, dirs_exist_ok=True)
            for key, body in ctx.items():
                if key.startswith("file:"):
                    rel = HT.normalize_rel(key[len("file:") :])
                    p = staged / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
            (staged / "skills").mkdir(exist_ok=True)
            (staged / "AGENTS.md").write_text(compose_agents_md(staged), encoding="utf-8")
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={"install_mode": INSTALL_MODE},
        )

    def run_env(self, base_env: Mapping[str, str], install: Install, model: str) -> dict:
        """The environment for one rollout of `install` under `model` (provider/model)."""
        staged = Path(install.staging_dir)
        return dsh_env(base_env, agents_home=staged, skills_dir=staged / "skills", model=model)

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "DshSubstrate was constructed without a launcher body; the benchmark adapter "
                "must supply one (it owns the workspace, the prompt and read-back locations)"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Checks DSH resolves and the substrate profile exists; starts nothing."""
        dsh_command()
        home = Path(os.environ.get("DSH_HOME") or Path.home() / ".dsh")
        prof = home / "profiles" / PROFILE / "cordis.patch.yml"
        if not prof.is_file():
            raise RuntimeError(
                f"DSH profile {PROFILE!r} missing ({prof}): create it with "
                f"`dsh --profile {PROFILE} --from-default-profile headless --help`, then copy "
                f"docs/dsh-h2.patch.yml over its cordis.patch.yml"
            )
        text = prof.read_text(encoding="utf-8", errors="replace")
        for var in ("H2_SKILLS_DIR", "H2_MODEL"):
            if var not in text:
                raise RuntimeError(
                    f"DSH profile {PROFILE!r} does not read {var}; copy docs/dsh-h2.patch.yml "
                    f"over {prof}"
                )


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def iter_events(path: Path):
    """The parseable event objects of one `headless --json` stream, in order."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            yield ev


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400) -> str:
    """Render DSH's `headless --json` event stream into the shared grammar.

    `final` repeats the last `text` event and is skipped; a turn that ends for any reason
    other than completion is recorded as an error, because that is how a stopped run looks.
    """
    if not Path(path).is_file():
        return f"(failed to read trajectory: {path} does not exist)"
    lines: list[str] = []
    for ev in iter_events(path):
        etype = ev.get("type")
        if etype == "text":
            txt = ev.get("text")
            if isinstance(txt, str) and txt.strip():
                lines.append(f"[ASSISTANT] {_clip(txt, obs_trunc)}")
        elif etype in ("reasoning", "thinking"):
            txt = ev.get("text") or ev.get("reasoning")
            if isinstance(txt, str) and txt.strip():
                lines.append(f"[THINKING] {_clip(txt, obs_trunc)}")
        elif etype == "tool_call":
            tin = ev.get("input")
            args = json.dumps(tin, default=str, ensure_ascii=False) if tin is not None else ""
            lines.append(f"[ACTION] {ev.get('tool') or 'tool'}({_clip(args, int(args_trunc))})")
        elif etype == "tool_result":
            out = ev.get("result")
            if not isinstance(out, str):
                out = json.dumps(out, default=str, ensure_ascii=False)
            if ev.get("status") not in (None, "completed"):
                lines.append(f"[OBSERVATION ERROR] {_clip(out or ev.get('error'), obs_trunc)}")
            else:
                lines.append(f"[OBSERVATION] {_clip(out, obs_trunc)}")
        elif etype == "error":
            msg = ev.get("message") or ev.get("error") or json.dumps(ev)[:300]
            lines.append(f"[ERROR] {_clip(msg, 300)}")
        elif etype == "status" and ev.get("phase") == "turn_end":
            reason = ev.get("reason") if isinstance(ev.get("reason"), dict) else {}
            if reason.get("kind") not in (None, "completed"):
                lines.append(f"[ERROR] turn ended: {_clip(json.dumps(reason), 300)}")
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


def final_text(path: Path) -> str:
    """The run's final assistant message, or "" when the stream has none."""
    text = ""
    for ev in iter_events(path):
        if ev.get("type") == "final" and isinstance(ev.get("text"), str):
            text = ev["text"]
    return text


def completed(path: Path) -> bool:
    """Whether the stream records a turn that ended by completing."""
    for ev in iter_events(path):
        if ev.get("type") == "status" and ev.get("phase") == "turn_end":
            reason = ev.get("reason") if isinstance(ev.get("reason"), dict) else {}
            if reason.get("kind") == "completed":
                return True
    return False


def token_usage(path: Path) -> dict[str, int]:
    """Summed per-step token usage; DSH reports tokens, not prices."""
    tot = {"input": 0, "output": 0, "cache_read": 0, "steps": 0}
    for ev in iter_events(path):
        if ev.get("type") == "status" and ev.get("phase") == "step_end":
            u = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            tot["input"] += int(u.get("inputTokens") or 0)
            tot["output"] += int(u.get("outputTokens") or 0)
            tot["cache_read"] += int(u.get("cacheReadTokens") or 0)
            tot["steps"] += 1
    return tot


__all__ = [
    "DSH_SURFACE",
    "INSTALL_MODE",
    "STREAM_NAME",
    "DshSubstrate",
    "compose_agents_md",
    "completed",
    "dsh_env",
    "dsh_tree_problems",
    "final_text",
    "install_scripts",
    "render_events",
    "run_headless",
    "token_usage",
]

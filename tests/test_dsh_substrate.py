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

"""The DSH substrate's contracts: staging, the event renderer, and the improver channel."""

import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2.core import prompts, tree
from harness2.core.proposer import Proposer
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec, resolve_channel_model
from harness2.substrate import dsh as DSH

EVENTS = [
    {"type": "session", "sessionId": "session-1", "cwd": "/w"},
    {"type": "status", "phase": "turn_start", "turn": 1},
    {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
    {"type": "text", "text": "Loading the skill."},
    {"type": "tool_call", "callId": "c1", "tool": "skill", "input": {"name": "ledger-sum"}},
    {"type": "tool_result", "callId": "c1", "status": "completed", "result": "<skill/>"},
    {
        "type": "status",
        "phase": "step_end",
        "turn": 1,
        "step": 1,
        "usage": {"inputTokens": 100, "outputTokens": 10, "cacheReadTokens": 50},
    },
    {"type": "tool_call", "callId": "c2", "tool": "pwsh", "input": {"command": "exit 1"}},
    {"type": "tool_result", "callId": "c2", "status": "error", "result": "boom"},
    {
        "type": "status",
        "phase": "step_end",
        "turn": 1,
        "step": 2,
        "usage": {"inputTokens": 20, "outputTokens": 5},
    },
    {"type": "text", "text": "Total: 85.15"},
    {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "completed"}},
    {"type": "final", "text": "Total: 85.15"},
]


def write_stream(path: Path, events=EVENTS) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def spec() -> BenchSpec:
    channel = ChannelSpec("dsh", "opencode-go/deepseek-v4-pro")
    return BenchSpec(
        name="t",
        runs_root=Path(tempfile.gettempdir()),
        install_mode=DSH.INSTALL_MODE,
        prompts=BenchPrompts(work="reconcile ledgers"),
        channels={"proposer": channel, "selector": channel},
    )


class DshSubstrateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def harness(self, extra: dict[str, str] | None = None) -> Path:
        body = {name: "" for name in DSH.DSH_SURFACE.always_on}
        body["systemprompt.md"] = (
            "Reconcile the ledger. When summing, load the `ledger-sum` skill.\n"
        )
        body["skills/ledger-sum/SKILL.md"] = (
            "---\nname: ledger-sum\ndescription: Sum a signed ledger\n---\nInclude negatives.\n"
        )
        body["scripts/check_total.py"] = "print('ok')\n"
        body["sub_agents/reader/AGENT.md"] = "---\ndescription: Read long files\n---\nRead.\n"
        body.update(extra or {})
        d = self.root / "harness"
        tree.write_tree(DSH.DSH_SURFACE, body, d, stubs=False)
        return d

    def test_materialise_composes_agents_md_and_scripts(self):
        sub = DSH.DshSubstrate()
        inst = sub.materialise(self.harness(), spec(), {})
        self.addCleanup(shutil.rmtree, inst.staging_dir, True)
        agents = (inst.staging_dir / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("load the `ledger-sum` skill", agents)
        self.assertIn("# AVAILABLE SUBAGENTS", agents)
        self.assertIn("`reader`", agents)
        self.assertIn("python scripts/<name>.py", agents)
        self.assertTrue((inst.staging_dir / "skills/ledger-sum/SKILL.md").is_file())
        ws = self.root / "ws"
        self.assertEqual(DSH.install_scripts(inst, ws), ["check_total.py"])
        self.assertTrue((ws / "scripts/check_total.py").is_file())
        env = sub.run_env({}, inst, "opencode-go/deepseek-v4-pro")
        self.assertEqual(env["DSH_AGENTS_HOME"], str(inst.staging_dir))
        self.assertEqual(env["H2_SKILLS_DIR"], str(inst.staging_dir / "skills"))
        self.assertEqual((env["H2_PROVIDER"], env["H2_MODEL"]), ("opencode-go", "deepseek-v4-pro"))

    def test_surface_has_no_agents_md_and_no_plugin(self):
        self.assertNotIn("AGENTS.md", DSH.DSH_SURFACE.always_on)
        self.assertIsNone(DSH.DSH_SURFACE.plugin_dir)
        self.assertFalse(tree.is_allowed(DSH.DSH_SURFACE, "AGENTS.md"))
        self.assertFalse(tree.is_allowed(DSH.DSH_SURFACE, "plugin/x.ts"))

    def test_non_kebab_skill_is_rejected(self):
        d = self.harness(
            {
                "skills/ledger_sum/SKILL.md": (
                    "---\nname: ledger_sum\ndescription: x\n---\nbody\n"
                ),
                "guardrails.md": "When summing, load the `ledger_sum` skill.\n",
            }
        )
        problems = DSH.dsh_tree_problems(tree.read_tree(DSH.DSH_SURFACE, d))
        self.assertEqual(len(problems), 1)
        self.assertIn("kebab-case", problems[0])
        with self.assertRaises(ValueError):
            DSH.DshSubstrate().materialise(d, spec(), {})

    def test_render_and_usage(self):
        stream = write_stream(self.root / "s.jsonl")
        text = DSH.render_events(stream, obs_trunc=1000, max_chars=10_000)
        self.assertEqual(
            text.splitlines(),
            [
                "[ASSISTANT] Loading the skill.",
                '[ACTION] skill({"name": "ledger-sum"})',
                "[OBSERVATION] <skill/>",
                '[ACTION] pwsh({"command": "exit 1"})',
                "[OBSERVATION ERROR] boom",
                "[ASSISTANT] Total: 85.15",
            ],
        )
        self.assertEqual(DSH.final_text(stream), "Total: 85.15")
        self.assertTrue(DSH.completed(stream))
        self.assertEqual(
            DSH.token_usage(stream), {"input": 120, "output": 15, "cache_read": 50, "steps": 2}
        )

    def test_stopped_turn_renders_as_error(self):
        events = EVENTS[:5] + [
            {"type": "status", "phase": "turn_end", "turn": 1, "reason": {"kind": "aborted"}}
        ]
        stream = write_stream(self.root / "s.jsonl", events)
        self.assertIn("[ERROR] turn ended", DSH.render_events(stream, obs_trunc=99, max_chars=999))
        self.assertFalse(DSH.completed(stream))
        missing = DSH.render_events(self.root / "nope.jsonl", obs_trunc=9, max_chars=9)
        self.assertIn("does not exist", missing)

    def test_prompts_have_a_dsh_surface(self):
        for pack in ("adapt", "tts"):
            bp = BenchPrompts(work="reconcile ledgers", prompt_pack=pack)
            text = prompts.propose(bp, "DOM", DSH.DSH_SURFACE)
            self.assertIn(
                "SKILL NAMES ARE KEBAB-CASE" if pack == "adapt" else "`subagent` tool", text
            )
            self.assertNotIn("plugin/<name>.ts", text)

    def test_channel_requires_provider_prefixed_model(self):
        self.assertEqual(resolve_channel_model("dsh", "opencode-go/x").kind, "dsh")
        with self.assertRaises(ValueError):
            resolve_channel_model("dsh", "deepseek-v4-pro")


FAKE_DSH = textwrap.dedent(
    """
    import json, os, pathlib, sys
    prompt = sys.stdin.read()
    assert "MISSION.md" in prompt, prompt
    home = pathlib.Path(os.environ["DSH_AGENTS_HOME"])
    assert home.is_dir() and not any(home.iterdir()), "improver must get an empty agents home"
    assert "H2_SKILLS_DIR" not in os.environ
    mission = pathlib.Path("MISSION.md").read_text(encoding="utf-8")
    h = pathlib.Path("context/harness/systemprompt.md")
    h.write_text(h.read_text(encoding="utf-8") + "Keep both signs.\\n", encoding="utf-8")
    pathlib.Path("output/answer.md").write_text("<notes>edited</notes>", encoding="utf-8")
    for e in ({"type": "text", "text": "done"}, {"type": "final", "text": "done"}):
        print(json.dumps(e))
    """
)


class DshChannelTests(unittest.TestCase):
    def test_improver_session_edits_the_harvested_tree(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        fake = tmp / "fake_dsh.py"
        fake.write_text(FAKE_DSH, encoding="utf-8")
        ctx = tmp / "ctx"
        (ctx / "harness").mkdir(parents=True)
        (ctx / "harness/systemprompt.md").write_text("Sum it.\n", encoding="utf-8")
        prop = Proposer(ChannelSpec("dsh", "opencode-go/deepseek-v4-pro"), audit_dir=tmp / "audit")
        with patch.object(DSH, "dsh_command", return_value=[sys.executable, str(fake)]):
            reply = prop.call("Edit the harness.", ctx, "inv1", harvest="context/harness")
        self.assertTrue(reply.ok, reply.error)
        self.assertEqual(reply.text, "<notes>edited</notes>")
        self.assertIn(
            "Keep both signs.", (reply.workspace / "systemprompt.md").read_text(encoding="utf-8")
        )

    def test_continuation_passes_the_session_id_before_stdin(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        fake = tmp / "argv_dsh.py"
        fake.write_text(
            "import json, sys\nsys.stdin.read()\nprint(json.dumps({'argv': sys.argv[1:]}))\n",
            encoding="utf-8",
        )
        stream = tmp / "s.jsonl"
        with patch.object(DSH, "dsh_command", return_value=[sys.executable, str(fake)]):
            rc, _, _ = DSH.run_headless("go on", tmp, {}, stream, timeout=60, session_id="s-1")
        self.assertEqual(rc, 0)
        argv = json.loads(stream.read_text(encoding="utf-8"))["argv"]
        self.assertEqual(
            argv,
            [
                "--profile",
                DSH.PROFILE,
                "--patch",
                str(DSH.overlay_path()),
                "--json",
                "--session-id",
                "s-1",
                "-",
            ],
        )


class DshIsolationTests(unittest.TestCase):
    def test_enabled_mcp_clients_are_reported(self):
        cfg = (
            "- id: llm\n  name: x\n"
            "- id: mcp-reason-network\n  name: '@deepseek-ai/dsh-mcp-client'\n  config: {}\n"
            "- id: mcp-other\n  name: '@deepseek-ai/dsh-mcp-client'\n  disabled: true\n"
        )
        self.assertEqual(
            DSH.isolation_problems(cfg),
            ["MCP client 'mcp-reason-network' is enabled in the effective DSH config"],
        )

    def test_harness_variables_do_not_reach_the_agent(self):
        env = DSH.dsh_env(
            {"PATH": "p", "HARNESS2_DSH_FORBIDDEN": "C:/x/tasks", "harness2_judge_api_key": "k"},
            agents_home=Path("a"),
            skills_dir=None,
            model="deepseek-official/deepseek-v4-pro",
        )
        self.assertEqual(env["PATH"], "p")
        self.assertFalse([k for k in env if k.upper().startswith("HARNESS2_")])

    def test_forbidden_paths_are_found_in_calls_and_results(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        events = [
            {
                "type": "tool_call",
                "tool": "read",
                "input": {"file_path": "C:\\b\\tasks\\x\\task.json"},
            },
            {"type": "tool_result", "status": "completed", "result": "nothing here"},
            {"type": "text", "text": "task.json is mentioned only in prose"},
        ]
        stream = write_stream(tmp / "s.jsonl", events)
        self.assertEqual(DSH.forbidden_touches(stream, ["task.json"]), ["read: task.json"])
        self.assertEqual(DSH.forbidden_touches(stream, ["C:\\b\\tasks"]), ["read: c:/b/tasks"])
        self.assertEqual(DSH.forbidden_touches(stream, ["scores.json"]), [])
        self.assertEqual(DSH.tool_call_count(stream), 1)


class DshCredentialTests(unittest.TestCase):
    def test_reads_only_the_refs_section(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / ".credentials.yaml").write_text(
            "version: 1\nrecords:\n  OPENCODE_GO_API_KEY:\n    kind: grant\n"
            "refs:\n  OTHER: nope\n  OPENCODE_GO_API_KEY: 'sk-test'\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"DSH_HOME": str(tmp)}):
            self.assertEqual(DSH.credential("OPENCODE_GO_API_KEY"), "sk-test")
            self.assertEqual(DSH.credential("MISSING"), "")


if __name__ == "__main__":
    unittest.main()

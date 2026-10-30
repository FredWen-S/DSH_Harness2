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

"""The LAB cell on DSH: its spec, runtime-owned names, and the compat judge environment."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks import lab
from harness2.substrate import dsh as DSH

HAVE_LAB = (config.lab_repo() / "harness" / "skills").is_dir()


@unittest.skipUnless(HAVE_LAB, "harvey-labs checkout not installed")
class LabDshTests(unittest.TestCase):
    def setUp(self):
        self.adapter = lab.LabAdapter(substrate="dsh")

    def test_spec_uses_the_dsh_channel_and_install_mode(self):
        spec = self.adapter.spec
        self.assertEqual(spec.install_mode, DSH.INSTALL_MODE)
        self.assertEqual(spec.channels["proposer"].kind, "dsh")
        self.assertIn("read_doc.py", spec.reserved_scripts)

    def test_runtime_skill_names_are_not_editable(self):
        tree = {
            "systemprompt.md": "When drafting Word files, load the `docx` skill.\n",
            "skills/docx/SKILL.md": "---\nname: docx\ndescription: x\n---\nbody\n",
        }
        with self.assertRaises(ValueError) as cm:
            self.adapter.validate_tree(tree)
        self.assertIn("runtime-provided", str(cm.exception))

    def test_delta_names_the_platform_shell(self):
        text = lab.DSH_ENV_DELTA.replace("{shell}", lab.DSH_SHELL_TOOL)
        self.assertIn(f"the shell tool is `{lab.DSH_SHELL_TOOL}`", text)
        self.assertIn("output/<filename>", text)
        self.assertIn("python scripts/read_doc.py", text)


class LabHelpersTests(unittest.TestCase):
    def test_forbidden_paths_cover_criteria_not_the_lab_interpreter(self):
        needles = lab._forbidden_needles()
        tasks = str((config.lab_repo() / "tasks").resolve())
        self.assertIn(tasks, needles)
        self.assertNotIn(str(config.lab_repo().resolve()), needles)
        venv_python = str(config.lab_repo().resolve() / ".venv" / "Scripts" / "python.exe")
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        stream = tmp / "s.jsonl"
        stream.write_text(
            "\n".join(
                json.dumps(ev)
                for ev in (
                    {"type": "tool_call", "tool": "pwsh", "input": {"command": "python -V"}},
                    {"type": "tool_result", "tool": "pwsh", "result": venv_python},
                    {"type": "tool_call", "tool": "read", "input": {"file_path": tasks}},
                )
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            DSH.forbidden_touches(stream, needles),
            [f"read: {tasks.lower().replace(chr(92), '/')}"],
        )

    def test_judge_id_strips_provider_prefixes(self):
        self.assertEqual(lab.judge_id("compat/gpt-6-luna"), "gpt-6-luna")
        self.assertEqual(lab.judge_id("vertex/claude-opus-4-8"), "claude-opus-4-8")
        self.assertEqual(lab.judge_id("gpt-5.5"), "gpt-5.5")

    def test_force_rmtree_removes_read_only_files(self):
        import os
        import stat
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp()) / "ws"
        (d / "documents").mkdir(parents=True)
        f = d / "documents" / "a.docx"
        f.write_text("x", encoding="utf-8")
        os.chmod(f, stat.S_IREAD)
        lab._force_rmtree(d)
        self.assertFalse(d.exists())


class CompatJudgeEnvTests(unittest.TestCase):
    def test_compat_judge_gets_endpoint_key_and_session_header(self):
        with (
            patch.object(DSH, "credential", return_value="sk-test"),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            for k in (
                "HARNESS2_JUDGE_BASE_URL",
                "HARNESS2_JUDGE_API_KEY",
                "HARNESS2_JUDGE_HEADERS",
            ):
                env.pop(k, None)
            got = lab._judge_env("compat/gpt-6-luna")
        self.assertEqual(got["HARNESS2_JUDGE_BASE_URL"], "https://opencode.ai/zen/go/v1")
        self.assertEqual(got["HARNESS2_JUDGE_API_KEY"], "sk-test")
        self.assertIn("x-opencode-session", json.loads(got["HARNESS2_JUDGE_HEADERS"]))

    def test_other_judges_get_the_plain_lab_env(self):
        got = lab._judge_env("vertex/claude-opus-4-8")
        self.assertNotIn(
            "HARNESS2_JUDGE_HEADERS", {k for k in got if k not in __import__("os").environ}
        )


if __name__ == "__main__":
    unittest.main()

# DeepSeek Harness (DSH) substrate

This fork adds [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) as a third
solver substrate (`--substrate dsh`) and a third improver channel (`dsh`), next to OpenCode and
Codex. The method (`harness2/core/`) is unchanged: recursion, prompts, budget ledger,
selection, reflection, library and judging all run as released. The fork also runs on Windows.

## Setup

1. Install DSH (`npm i -g @deepseek-ai/dsh`) and store a provider key in it (the web Models
   page writes `~/.dsh/.credentials.yaml`). The defaults use OpenCode Go
   (`OPENCODE_GO_API_KEY`).
2. Create the substrate profile and install its patch layer:

   ```bash
   dsh --profile h2 --from-default-profile headless --help
   cp docs/dsh-h2.patch.yml ~/.dsh/profiles/h2/cordis.patch.yml
   ```

   The layer re-enables local credentials (the headless template reads keys from the
   environment only), makes the executor model overridable per run, and restricts skill
   discovery to the one root each rollout stages.
3. `uv venv --python 3.12 .venv && uv pip install -e '.[render]'`

Models are `provider/model` ids as DSH names them:

| variable | default | role |
|---|---|---|
| `HARNESS2_DSH_SOLVER_MODEL` | `opencode-go/deepseek-v4-pro` | rollouts |
| `HARNESS2_DSH_IMPROVER_MODEL` | `opencode-go/deepseek-v4-pro` | editor, composer, selector, reflector |
| `HARNESS2_DSH_PROFILE` | `h2` | profile the runner boots |
| `HARNESS2_DSH_BIN` | `dsh` on PATH | launcher override |

## How a harness reaches DSH

| component | editable path | at runtime |
|---|---|---|
| c1-c4 always-on | `systemprompt.md`, `guardrails.md`, `LongTermMEMORY.md`, `tool_descriptions/tool_guidance.md` | composed into `AGENTS.md` in a per-rollout directory that DSH adopts as `DSH_AGENTS_HOME`; DSH's agent-instructions plugin injects it as the user-global chain |
| c5 subagents | `sub_agents/<name>/AGENT.md` | advertised in `AGENTS.md`; the agent passes the role to DSH's generic `subagent` tool (DSH presets are host bundle rows, not per-run files) |
| c6 skills | `skills/<name>/SKILL.md` (+ `scripts/`) | the only skill root DSH scans (`H2_SKILLS_DIR`); loaded on demand by the `skill` tool. Names must be kebab-case, which validation enforces |
| c7 scripts | `scripts/<name>.py` | copied to the workspace's `scripts/` |
| c8 plugins | not on the surface | DSH plugins are host packages; the surface omits c8 rather than read candidate plugins as prose |

The always-on files keep OpenCode's names on purpose: improver sessions also run under DSH,
which loads any `AGENTS.md` its tools read into, so a harness file named `AGENTS.md` would be
applied to the editor editing it.

Each rollout is one `dsh --profile h2 --json -` process with the task on stdin and the
workspace as cwd, which is also the sandbox's writable root (`workspace-write`). The event
stream is saved as `dsh_session.jsonl` and rendered into the shared trajectory grammar.

## Live demo

```bash
harness2-dsh-demo --output runs/dsh-demo --k 1 --mode parallel
```

Three bookkeeping tasks whose conventions live in each workspace's `NOTES.md`. Everything
is a real DSH turn; the expected totals exist only in `harness2/dsh_demo.py` and are compared
after selection.

## LAB on DSH

The DSH cell mirrors the OpenCode cell: the same base prompt body, the same
`results/<run_id>/{output,metrics.json,config.json}` layout `run_eval` scores, the base skill
manuals (`docx`, `xlsx`, `pptx`) as runtime skills, `scripts/read_doc.py` with host parsing,
and `run_opencode.py`'s continuation policy (up to two nudges, skipped only when a deliverable
exists and the last reply is exactly `DONE`, stopped by a short budget or a nudge that made no
tool call). A DSH environment delta (absolute paths, explicit `output/`, `read_doc.py`, the
platform's shell tool) is appended to `AGENTS.md` at launch, as `ENV_DELTA` is for OpenCode;
it is runtime, not harness.

```bash
harness2-setup lab               # or: clone + patches + overlays + uv sync, see below
export HARNESS2_JUDGE_MODEL=compat/gpt-6-luna
harness2 --bench lab --substrate dsh --k 1 --mode parallel --split test --split-file <split.json>
python scripts/score_base_vs_final.py --bench lab --substrate dsh --namespace lab-dsh-agg1
```

`compat/<model>` is an OpenAI-compatible chat judge added by
`third_party/patches/harvey-labs-judge-openai-compat.diff`. It reads
`HARNESS2_JUDGE_BASE_URL` (default OpenCode Go), `HARNESS2_JUDGE_API_KEY` (default: the
DSH-stored `OPENCODE_GO_API_KEY`) and `HARNESS2_JUDGE_HEADERS`; OpenCode Go additionally needs
an `x-opencode-session` header, which the adapter supplies. Changing the judge changes the
measurement: these numbers are not comparable with the paper's Vertex-judged tables.

On Windows: enable `core.longpaths` in the harvey-labs checkout (some task paths exceed 260
characters), and if pandoc is not installed, `uv pip install pypandoc_binary` into the LAB
venv; the adapter puts its bundled binary on the rollout's PATH.

## Isolation

What a run boots is checked, not assumed:

- Every DSH process gets `--patch` with `ISOLATION_OVERLAY` (`harness2/substrate/dsh.py`),
  which applies after the profile **and** the user-global `~/.dsh/cordis.patch.yml`. It
  disables the reason-network MCP client that layer inserts; preflight dumps the effective
  config and refuses to start while any MCP client is still enabled.
- The digest of that effective config and the permission mode (`runtime_fingerprint`) is part
  of the sweep identity, so a resume under a changed DSH setup fails instead of mixing.
- **Reads are not confined.** DSH's Windows sandbox enforces writes only (`partial`), so a
  rollout or improver session could read the benchmark checkout. Mitigations: LAB rollout
  workspaces live outside the repository (`HARNESS2_DSH_WORKROOT`, default `%TEMP%/h2ws`),
  rollouts get no `PYTHONPATH` and a parser-only `LAB_BENCH_ROOT`, and every DSH stream is
  scanned for tool calls or results that mention ground-truth or judge paths
  (`HARNESS2_DSH_FORBIDDEN`). An improver session that touches one is discarded; a rollout
  that does is recorded in `metrics.json` `forbidden_access` and reported by
  `scripts/score_base_vs_final.py`. This is detection, not a boundary.
- Judge verdicts are keyed by judge: a cached `scores.json` from another judge is set aside
  and re-scored, and the comparison script refuses to mix judges.

## Differences from the released substrates

- No c8: candidate middleware cannot be executed on DSH.
- Subagents are role prompts for the generic `subagent` tool, not named presets.
- DSH reports tokens, not prices, so `usd` fields are 0; `metrics.json` carries token counts.
- The judge in the DSH LAB recipe is an OpenCode Go model, not the paper's Vertex judge.

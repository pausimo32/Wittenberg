# Setup — Wittenberg on any computer, CPU only

Environment this file assumes: **Windows, VSCode with the Claude Code extension, no WSL.** The
app itself needs **no GPU on any machine**. See `docs/adr/0003` for why the framework is PyTorch, and `docs/adr/0005` for why the app
needs no GPU at all: inference is ≈ 12 ms on CPU and the checkpoint is 1.17 MB.

Training is the only step that wants a GPU. It runs on the team's RTX 5070 a handful of times,
on that machine only, and is not part of the app (`docs/00-master-plan.md`, F4). See §5b.

## 0. Prerequisites

| | Check |
|---|---|
| Claude Code | extension ≥ 2.1.219 (Opus 5 needs it). The version is the folder name under `%USERPROFILE%\.vscode\extensions\anthropic.claude-code-*` |
| Python 3.11 | `python --version` |
| Node 20 | `node --version` |
| Git | `git --version` |

**The CLI is optional.** `claude --version` failing because there is no binary on PATH is
expected when only the VSCode extension is installed, and nothing in this project needs the
CLI. Install it (`npm i -g @anthropic-ai/claude-code`) only if you later want headless runs —
for example `claude -p` to batch the F4 ablations overnight.

**There is no `make`.** The task runner is `tasks.py`. Everything is `python tasks.py <target>`;
`python tasks.py --list` shows them all.

## 1. Drop the kit into the repo

```
CLAUDE.md            PROMPTS.md          SETUP.md            tasks.py
requirements.txt     config/preprocess.json   config/versions.json
docs/00-master-plan.md … docs/07-glossary-ca-en.md
docs/adr/0001-…  0002-…  0003-…  0004-…
.claude/agents/wittenberg-{repro-auditor,dataset-curator,tdr-writer}.md
.claude/skills/{dataset-ingest,tdr-annex-export}/SKILL.md
```

`git init && git add -A && git commit` before running anything else.

## 2. Plugins — the one step no agent can do for you

`/plugin` is an interactive slash command. It cannot be run from Bash, from a skill, or by a
subagent. Paste these six lines into the Claude Code chat yourself:

```
/plugin install document-skills@anthropic-agent-skills
/plugin install superpowers@claude-plugins-official
/plugin install pyright-lsp@claude-plugins-official
/plugin install typescript-lsp@claude-plugins-official
/plugin install code-review@claude-plugins-official
/plugin install frontend-design@claude-plugins-official
```

Both marketplaces (`anthropic-agent-skills`, `claude-plugins-official`) are normally already
registered in `~/.claude/settings.json` — check before adding them. If one is missing:

```
/plugin marketplace add anthropics/skills
/plugin marketplace add anthropics/claude-plugins-official
```

Verify afterwards: `enabledPlugins` in `~/.claude/settings.json` should list all six. `null`
means nothing installed.

## 3. Import the four external subagents

Copy individual files. Do **not** install whole collections — every agent's description costs
context on every request, permanently.

```powershell
$tmp = "$env:TEMP\wb-agents"
git clone --depth 1 https://github.com/wshobson/agents $tmp\wshobson
git clone --depth 1 https://github.com/rohitg00/awesome-claude-code-toolkit $tmp\rohitg00
git clone --depth 1 https://github.com/VoltAgent/awesome-claude-code-subagents $tmp\voltagent

copy $tmp\wshobson\plugins\machine-learning-ops\agents\ml-engineer.md    .claude\agents\
copy $tmp\wshobson\plugins\machine-learning-ops\agents\data-scientist.md .claude\agents\
copy $tmp\rohitg00\agents\data-ai\computer-vision-engineer.md            .claude\agents\
copy $tmp\voltagent\categories\02-language-specialists\react-specialist.md  .claude\agents\
```

Clone outside the repo. Verify with `git status` that nothing was cloned inside it.

## 4. Frontmatter hygiene — check every imported file

**a) `name` must be what you think it is.** Claude Code resolves agents by the `name` field,
not by the filename. In `wshobson/agents` the two do not always match: `ml-engineer.md`
declares `name: ml-engineer`, but `code-reviewer.md` declares
`name: comprehensive-review-code-reviewer`. The four files above happen to match — verify
anyway, and re-verify after any re-import.

**b) No pinned legacy model IDs.** A hardcoded `model: claude-sonnet-4-20250514` silently runs
the agent on an obsolete model. Replace it with a tier name or `inherit`.

Tier pins (`opus`, `sonnet`, `haiku`) are **not** legacy IDs and should be left alone. The rule:

> Pin when the agent's job has a stable difficulty of its own. Use `inherit` when the agent is
> a general helper whose difficulty tracks the parent task.

So `computer-vision-engineer: opus` stays (it works F2, where errors are expensive) and a
frontend specialist stays on `sonnet` (F6 and F7 implement a written spec, and the pin keeps it
cheap even when the main session is on Opus). The three `wittenberg-*` agents are pinned for the
same reason. If you exhaust the Opus quota mid-phase, flip the opus-pinned agents to `inherit`
temporarily and carry on with Sonnet.

**If you already imported `nextjs-developer`**, replace it with `react-specialist` from the same
repository — the app is Vite + React now, not Next.js (`docs/adr/0004`). Delete the old file;
leaving both means two overlapping descriptions competing for the same context budget.

**c) Descriptions ≤ ~300 characters.** They live in context permanently. Shorten while keeping
the trigger words.

**d) Permission policy — permanent rule for this repo.**

> No file in `.claude/agents/` may declare `permissionMode` with the value
> `bypassPermissions`, `dontAsk` or `auto`. If an imported agent carries one, delete the line
> (the default is `default`) and report it. The only values allowed here are `default` and
> `plan`.
>
> Reason: these agents run against a repository that will contain photographs taken by minors
> and credentials for the study questionnaires. No third-party agent runs actions unconfirmed.

## 5. Python and Node environment

```powershell
python tasks.py setup
```

That creates `.venv`, installs `requirements.txt`, runs `npm install`, and then asserts two
things that are easy to get silently wrong:

- **OpenCV is the pinned version.** Not for cross-language parity — there is only one
  implementation now — but so that all three of you, and a reader reproducing this in a year,
  get the same pixels (`docs/03-preprocessing-spec.md` §5.3).
- **The device is reported.** `python tasks.py check-device` prints `cuda: False` on a machine
  without a GPU, and that is a **PASS**. CPU is the supported configuration: ≈ 24 ms
  preprocessing + ≈ 12 ms inference per image.

Do not install CUDA wheels. They weigh hundreds of megabytes, the app never uses them, and they
make the repository fail to install on the next computer you try it on — which is the one thing
this setup is designed to avoid.

## 5b. The training machine only — RTX 5070

Skip this on any computer that just runs the app; it needs nothing but `requirements.txt`.

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt
python tasks.py check-device
```

`check-device` must print the device **and** `gpu op OK`. It runs a real matrix multiply on
purpose: RTX 50-series is Blackwell (sm_120), and `cu121`/`cu124` PyTorch builds report
`cuda: True`, print the card's name correctly, and then fail on the first actual operation with
`no kernel image is available for execution on the device`. No driver update fixes that — the
build has to be **cu128 or newer**. Check the current index URL at
<https://pytorch.org/get-started/locally/> before installing.

## 6. Verify Claude Code

```
/agents     # 7: the three wittenberg-* plus the four imported
/context    # record the baseline cost of CLAUDE.md + skill descriptions
/doctor     # skill listing budget
```

If the Skills row in `/context` is large, add `disable-model-invocation: true` to skills you
only ever invoke by hand — that drops their description cost to zero.

## 7. Session defaults and hygiene

```
/model opus          # architecture and specs
/effort high         # xhigh only for reproducibility audits
```

Switch to `/model sonnet` for implementation phases. On Max, Opus has its own quota separate
from the shared session and weekly limits, so hitting the Opus wall does not stop you — switch
model and continue. `fable` may bill to usage credits depending on plan and seat tier; the
`/model` picker marks it. Default fan-outs to `sonnet`.

- `/clear` between phases. Phase boundaries are context boundaries.
- Re-read `PROMPTS.md` before writing a prompt for a new phase.
- Once a skill is invoked its full text stays in context for the rest of the session — invoke
  deliberately, not to see what it says.

## 8. Windows gotchas that will cost you an afternoon

| Symptom | Cause |
|---|---|
| `UnicodeDecodeError` reading a `.md` or `.csv` | Windows still defaults to cp1252 in places. Pass `encoding="utf-8"` on **every** open/read/write. |
| `OSError: symbolic link privilege not held` | Something tried `os.symlink`. This repo uses `runs/latest.txt`, a pointer file. Never symlink. |
| A script works for one team member, not another | a shell-only utility (`grep`, `sed`, `rm`) leaked into a script. Keep those out of anything the team runs. |
| Paths break with long project names | enable long paths: `git config --system core.longpaths true`. |
| `pillow-heif` fails to import | install the wheel, not a source build; confirm the Python version matches the wheel. |

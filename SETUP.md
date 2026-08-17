# Setup — installing the Claude Code environment for Wittenberg

Do this once, before phase F0. Requires Claude Code v2.1.219+ (Opus 5 is not offered by older
versions). Check with `claude --version`.

## 1. Drop the kit into the repo

```
CLAUDE.md
PROMPTS.md
SETUP.md
config/preprocess.json
docs/00-master-plan.md … docs/07-glossary-ca-en.md
docs/adr/0001-…, 0002-…
.claude/agents/wittenberg-{parity-auditor,dataset-curator,tdr-writer}.md
.claude/skills/{preprocessing-parity,dataset-ingest,tdr-annex-export}/SKILL.md
```

Commit before running anything.

## 2. Install plugins

Run inside `claude` in the repo root.

```
/plugin marketplace add anthropics/skills
/plugin install document-skills@anthropic-agent-skills

/plugin install superpowers@claude-plugins-official
/plugin install pyright-lsp@claude-plugins-official
/plugin install typescript-lsp@claude-plugins-official
/plugin install code-review@claude-plugins-official
/plugin install frontend-design@claude-plugins-official
```

`claude-plugins-official` is added automatically on first interactive start. If it is missing:
`/plugin marketplace add anthropics/claude-plugins-official`.

## 3. Install the third-party subagents

These are community repos. Copy only the agent files you need into `.claude/agents/` rather
than installing whole collections — every agent's description sits in context permanently.

```bash
mkdir -p /tmp/ext && cd /tmp/ext

git clone --depth 1 https://github.com/wshobson/agents
cp agents/plugins/machine-learning-ops/agents/ml-engineer.md   <repo>/.claude/agents/
cp agents/plugins/machine-learning-ops/agents/data-scientist.md <repo>/.claude/agents/

git clone --depth 1 https://github.com/rohitg00/awesome-claude-code-toolkit
cp awesome-claude-code-toolkit/agents/data-ai/computer-vision-engineer.md <repo>/.claude/agents/

git clone --depth 1 https://github.com/VoltAgent/awesome-claude-code-subagents
cp awesome-claude-code-subagents/categories/02-language-specialists/nextjs-developer.md \
   <repo>/.claude/agents/
```

**Then open each copied file and check its `name:` field.** In `wshobson/agents` the frontmatter
name does not reliably match the filename — some are plugin-prefixed
(`comprehensive-review-code-reviewer`), some are not (`ml-engineer`). Claude Code resolves
agents by `name`, not by path.

Also check for a hardcoded legacy `model:` (e.g. `claude-sonnet-4-20250514`) and change it to
`inherit` or `sonnet`, or the agent will silently run on an old model.

## 4. Verify

```
/agents          # the three wittenberg-* plus the four imported ones
/context         # baseline cost of CLAUDE.md + skill descriptions
/doctor          # skill listing budget
```

Budget check: if the Skills row in `/context` is large, the cheapest fix is
`disable-model-invocation: true` on skills you only ever invoke by hand — that drops their
description cost to zero.

## 5. Session defaults

```
/model opus          # architecture and specs
/effort high         # xhigh only for parity audits
```

Switch to `/model sonnet` for implementation phases. On Max, Opus has its own quota separate
from the shared session and weekly limits, so hitting the Opus wall does not stop you working —
switch model and continue.

Before fanning out subagents, know that `fable` may bill to usage credits depending on plan and
seat tier; the `/model` picker marks it. Default fan-outs to `sonnet`.

## 6. Hygiene

- `/clear` between phases. Phase boundaries are context boundaries.
- Re-read `PROMPTS.md` before writing a prompt for a new phase.
- Once a skill is invoked its full text stays in context for the rest of the session — invoke
  deliberately, not to see what it says.

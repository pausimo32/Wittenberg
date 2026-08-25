# Wittenberg — prompt template for Claude Code

## The template

Copy, fill the six blocks, delete what does not apply. Order matters: context before task,
task before constraints, constraints before the definition of done.

```
## Context
Phase <Fn> of the master plan. <One sentence of state: what exists now.>
Read: <docs/NN-*.md>   ← at most 2. If you need 3, the task is too big.
Do NOT read: <the rest of docs/>.

## Task
<One imperative sentence. One outcome.>

## Constraints
- <spec section that governs this, by file and §>
- <what must not change>
- <performance / size / determinism budget>

## Delegate
- <subagent-name> for <the part that reads many files>
- Use the <skill-name> skill for <the procedural part>

## Definition of done
- [ ] <a command that exits 0>
- [ ] <a file that exists>
- [ ] <a number that meets a threshold>

## Model
<opus | sonnet | haiku | fable> — <one clause of justification>
```

## Why each block exists

**Context** puts the doc budget in the prompt instead of hoping Claude guesses. Naming what
*not* to read is as valuable as naming what to read — without it, a broad task pulls in every
spec file and burns the context window before work starts.

**Task** is one sentence because a prompt with three tasks produces three half-finished ones.

**Constraints** cite spec sections rather than restating them. `docs/03-preprocessing-spec.md §2`
costs eight tokens; pasting the pipeline costs six hundred and creates a second source of truth
that will drift.

**Delegate** is the token lever. A subagent reads twenty files in its own context and returns a
paragraph. Without this block, those twenty files land in the main window and stay there for
the session.

**Definition of done** must be machine-checkable. "Works correctly" is not a gate;
`python tasks.py eval` exiting 0 and writing `metrics.json` into the directory named by
`runs/latest.txt` is.

**Model** — see the table below. Stating it stops the default from being wrong silently.

## Model selection

| Model | Use for | Why |
|---|---|---|
| `opus` | architecture, spec changes, reproducibility audits, TDR prose, ablation design | deep reasoning; the mistakes here are expensive and slow to detect |
| `sonnet` | implementation from a written spec, tests, refactors, UI components | the workhorse; a good spec makes the reasoning cheap |
| `haiku` | renaming, formatting, mechanical edits, log grepping | no reasoning needed; fastest and cheapest |
| `fable` | a phase-sized job in one go (F4 end-to-end), or a bug surviving two sonnet attempts | long-horizon; **check whether it bills usage credits on your plan first** |

**Pinning vs `inherit` in a subagent.** Pin the model when the agent's job has a stable
difficulty of its own (`wittenberg-repro-auditor` is always hard; a frontend specialist is always
implementation). Use `inherit` when the agent is a general helper whose difficulty tracks the
parent task. If you run out of Opus quota mid-phase, temporarily flip the opus-pinned agents to
`inherit` and keep working on Sonnet.

On Max, Opus has a separate quota from the shared session/weekly limits, so switching to
`sonnet` when you hit the Opus wall keeps you working. Fan-outs default to `sonnet`.

`/effort xhigh` for reproducibility audits and architecture. `high` (the default) otherwise.

## Token discipline — the rules that actually matter

1. **Two docs maximum per task.** More means the task is under-decomposed. Split it.
2. **Cite sections, do not paste them.** `§4` not the table.
3. **Delegate anything that reads more than ~5 files.** Subagent context is separate; only the
   summary returns.
4. **`/clear` between phases.** Phase boundaries are natural context boundaries — F4's work is
   irrelevant to F6.
5. **An invoked skill stays in context for the rest of the session.** Invoke deliberately, not
   exploratorily.
6. **`/context` before a long session** to see what CLAUDE.md and skill descriptions already cost.
7. Keep CLAUDE.md under ~120 lines. It is re-sent on every single request; a paragraph added
   there is paid for hundreds of times.

## Worked example

Phase F5, the local serving API:

```
## Context
Phase F5. src/preprocess.py is done and tested. wittenberg_v1 is trained and
runs/latest.txt points at a passing run. No API exists yet.
Read: docs/05-webapp-spec.md.
Do NOT read: docs/01, 02, 03, 04, 06, 07.

## Task
Implement the FastAPI backend described in §3.

## Constraints
- /api/predict MUST call src.preprocess.preprocess directly. Do not
  reimplement, inline, wrap or "optimise" any transform step. This is the
  property the whole architecture exists to guarantee (docs/adr/0004).
- Assert at startup that the checkpoint's preprocess_hash equals the SHA-256
  of the current config/preprocess.json. On mismatch, refuse to start and
  print both. Never downgrade this to a warning.
- Bind 127.0.0.1 explicitly. No CORS wildcard. Images processed in memory,
  never written to disk.
- Load the model once at startup, .to('cuda').eval(). Single worker.
- EXIF/HEIC/P3 normalisation comes from the shared src/imageio.py helper —
  the same one ingest uses. Do not write a second copy (spec §5.1).

## Definition of done
- [ ] python tasks.py check-model exits 0
- [ ] pytest: /api/predict on a tests/fixtures/ image returns exactly what calling
      preprocess + model directly returns
- [ ] pytest: mutating config/preprocess.json makes startup fail
- [ ] /api/health reports run id, preprocess version, device = cuda
- [ ] elapsed_ms < 200 on the local GPU

## Model
sonnet — the spec is complete; this is assembly. Escalate to opus only if the
startup guard design turns out to be ambiguous.
```

## Anti-patterns

| Do not | Instead |
|---|---|
| "Improve the preprocessing" | name the step, the parameter, and the metric that should move |
| "Make the model better" | one ablation from `docs/04-model-spec.md` §5, with its acceptance gate |
| "Deploy it" / "make it work on phones" | out of scope — the app is local-only (`docs/adr/0004`) |
| "Read all the docs and then..." | name the two docs |
| "Make the startup check pass" | fix the mismatch or retrain — weakening a guard is the failure mode |
| pasting a spec table into the prompt | cite the section |
| a prompt spanning two phases | one prompt per phase |

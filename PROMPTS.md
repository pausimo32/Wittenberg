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
<opus | sonnet | haiku> — <one clause of justification>
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

**Definition of done** must be machine-checkable. "Works correctly" is not a gate; `make eval &&
test -f runs/latest/metrics.json` is.

**Model** — see the table below. Stating it stops the default from being wrong silently.

## Model selection

| Model | Use for | Why |
|---|---|---|
| `opus` | architecture, spec changes, parity debugging, TDR prose, ablation design | deep reasoning; the mistakes here are expensive and slow to detect |
| `sonnet` | implementation from a written spec, tests, refactors, UI components | the workhorse; a good spec makes the reasoning cheap |
| `haiku` | renaming, formatting, mechanical edits, log grepping | no reasoning needed; fastest and cheapest |
| `fable` | a phase-sized job in one go (F4 end-to-end), or a bug surviving two sonnet attempts | long-horizon; **check whether it bills usage credits on your plan first** |

On Max, Opus has a separate quota from the shared session/weekly limits, so switching to
`sonnet` when you hit the Opus wall keeps you working. Fan-outs default to `sonnet`.

`/effort xhigh` for parity audits and architecture. `high` (the default) otherwise.

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

```
## Context
Phase F6. Python preprocessing (src/preprocess.py) is done and its unit tests pass.
The browser side does not exist yet.
Read: docs/03-preprocessing-spec.md, docs/05-webapp-spec.md.
Do NOT read: docs/01, 02, 04, 06, 07.

## Task
Implement web/lib/preprocess.ts so it reproduces src/preprocess.py exactly.

## Constraints
- Follow docs/03-preprocessing-spec.md §2, all 8 steps, in order, no fusion.
- All parameters read from config/preprocess.json. Zero numeric literals in the transform code.
- opencv.js only. No canvas 2D, no drawImage, no hand-written pixel loops (§5.1, §5.2).
- Runs in a Web Worker. Must not block the main thread.
- Do not modify src/preprocess.py.

## Delegate
- wittenberg-parity-auditor once a first version compiles.
- Use the preprocessing-parity skill to run and read the gate.

## Definition of done
- [ ] npm run parity exits 0
- [ ] every stage meets its §4 threshold; report the actual numbers
- [ ] no numeric literal in the transform path (grep it)
- [ ] tsc --noEmit clean

## Model
sonnet — the spec is complete, this is transcription. Escalate to opus only if parity fails
for a reason not listed in §5.
```

## Anti-patterns

| Do not | Instead |
|---|---|
| "Improve the preprocessing" | name the step, the parameter, and the metric that should move |
| "Make the model better" | one ablation from `docs/04-model-spec.md` §5, with its acceptance gate |
| "Read all the docs and then..." | name the two docs |
| "Fix the parity test" | "find the first divergent stage" — fixing the test is the failure mode |
| pasting a spec table into the prompt | cite the section |
| a prompt spanning two phases | one prompt per phase |

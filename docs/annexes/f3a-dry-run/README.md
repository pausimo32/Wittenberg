# F3a dry run — the archived record of a dataset that was never real

**Generated:** 2026-08-25, by `tools/synth_capture.py --sessions 4 --frames 20 --classes 6`
**Environment:** `opencv-contrib-python` 4.12.0.88 (cv2 4.12.0), Python 3.11.9, numpy 2.1.3
**Preprocessing:** `config/preprocess.json` 1.1.0
**Seed:** 42, everywhere

This directory is **evidence, not input**. Nothing in the project reads it. It exists so the
TDR can show that the ingest pipeline and the training bench were built, exercised and greened
*before a single real photograph existed* — which is the whole argument of F2.5, F3a and F4a,
and is not a claim anyone should have to take on trust.

## What is here

| File | What it is |
|---|---|
| `modes.csv` | 8 invented modes, 6 of them usable — the close pair at 2760/2785 Hz exists so the 40 Hz separation rule (`docs/02` §3) actually fires |
| `classes.json` | the frozen 6-class index built from `modes.csv` by `tasks.py modes` |
| `metadata.csv` | 480 rows, one per rendered frame, 348 accepted |
| `ingest_report.json` | the yield, the `config/quality_gate.json` thresholds verbatim, and every rejected frame with its measurements |
| `calib/` | the four session calibrations, ArUco corners and px/mm |

Four sessions, `SYNTH-20260101-01` through `SYNTH-20260104-01`, on four consecutive days, in a
2 × 2 factorial of material × lighting so that no condition predicts the split. 27.5 % of frames
were rejected, and they were rejected on purpose: `tools/synth_capture.py` spoils a fixed
fraction of frames, one per §6 rejection reason, so the yield report has something real to count.

## Why none of these numbers may be reported

The images come from a closed-form square-plate mode shape. A CNN reaches ~100 % on them — both
architectures did — and that figure measures how well the network inverted the formula that drew
the pictures. It says nothing whatsoever about a vibrating aluminium plate.

The rejection rate is the same kind of number: it describes the renderer's deliberate defect
schedule, not a capture session. Research question S4 is answered from the **real**
`data/ingest_report.json`, once F3b has produced one.

What this run does legitimately demonstrate, and what the TDR may say:

- the ingest pipeline, the quality gate, session-wise splitting and the curator audit all ran
  end to end before the lab work began;
- the condition-confound trap in `docs/00-master-plan.md` F3b was found here, on rendered data,
  rather than after four real capture sessions had been shot;
- the training bench writes every artefact `docs/04` §3 and §4 require, and `check-model` gates
  the checkpoint.

## Why it was moved out of the live tree

`data/metadata.csv` and `config/classes.json` were sitting in the paths the pipeline reads.
Both were harmful there:

- **`metadata.csv`** would make the first real ingest fail with `MixedProvenanceError`. That is
  the guard doing its job, but a guard that has to be defeated by hand on the first day of real
  work is a guard people learn to defeat. Archiving it means the first real ingest simply runs.
- **`classes.json`** is worse. Six invented frequencies sitting in the live config while F1 goes
  and measures the real ones is a trap waiting for whoever forgets to regenerate it: the class
  index is frozen and is what a checkpoint stores instead of a frequency, so training against a
  stale list produces a model that answers with the wrong frequency at full confidence.
  **The live `config/classes.json` is deliberately absent. F1 creates it.**
- **`data/SYNTHETIC`** was deleted rather than archived. Its only job was to mark the *live*
  tree as synthetic, and the live tree no longer is. `tools/synth_capture.py` writes it again if
  anyone runs the dry run a second time.

`data/raw/SYNTH-*`, `data/processed/` and the synthetic `runs/` were deleted outright: they are
large, they are gitignored, and `tools/synth_capture.py` reproduces all of them from seed 42.

## Reproducing it

```bash
python tools/synth_capture.py --sessions 4 --frames 20 --classes 6
python tasks.py modes
python tasks.py ingest --all
python tasks.py split
python tasks.py train
python tasks.py eval
```

Then delete `data/raw/SYNTH-*`, `data/processed/`, `data/SYNTHETIC`, `data/metadata.csv`,
`config/classes.json` and the runs before touching real captures. The generator is seeded, so
the frames come out byte-identical under the pinned OpenCV — and only under the pinned OpenCV
(`docs/adr/0006`).

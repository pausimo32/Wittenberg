# ADR 0006 — The OpenCV baseline is `opencv-contrib-python` 4.12, and it is not hygiene

**Status:** accepted, 2026-08-25
**Supersedes nothing. Required by** `docs/03-preprocessing-spec.md` §5.3a, which says a
version pin that is resolved must be recorded in an ADR with the reason.

## Context

`config/versions.json` pinned `opencv-contrib-python` 4.12.0 while the working environment
had 5.0.0 on Python 3.14. §5.3a offers exactly two resolutions — move the pin to the
environment, or move the environment to the pin — and forbids the third one, which is to
leave `check-versions` red and continue.

The environment was moved to the pin. The argument for it *at the time* was hygiene: no
dataset and no checkpoint existed yet, so aligning cost nothing, and §5.3a says do it now
rather than after F3b.

Then it was measured, and the measurement is a better reason than the argument was.

## The measurement

`tools/opencv_baseline_diff.py`, over the 120 frames of `data/raw/SYNTH-20260101-01`,
`config/preprocess.json` at 1.1.0. Both environments on **Python 3.11.9 and numpy 2.1.3**,
so the only variable is OpenCV. (Running the 5.0.0 side on Python 3.14.4 / numpy 2.5.2
instead gives byte-identical output to the 3.11 run, which is what licenses attributing all
of the following to OpenCV alone.)

| | pinned | compared against |
|---|---|---|
| distribution | `opencv-contrib-python` 4.12.0.88 | `opencv-contrib-python` 5.0.0.93 |
| `cv2.__version__` | 4.12.0 | 5.0.0 |

### Where the difference comes from — one operation

Each stage fed the **pinned** environment's output of the previous stage, so every row is
that operation's own contribution rather than the sum of everything before it:

| step | stage | max abs difference | frames differing |
|---|---|---|---|
| 2 | `roi` — `warpPerspective`, `INTER_LINEAR` | **7 grey levels** | **120 / 120** |
| 3 | `gray` — `cvtColor` | 0 | 0 / 120 |
| 4 | `clahe` — `createCLAHE().apply` | 0 | 0 / 120 |
| 5 | `blur` — `GaussianBlur` | 0 | 0 / 120 |
| 6 | `edges` — `Canny` | 0 | 0 / 120 |
| 7 | `resized` — `resize`, `INTER_AREA` | 0 | 0 / 120 |
| 8 | `tensor` — scale | 0 | 0 / 120 |

**`cv2.warpPerspective` is the whole of it.** CLAHE, Canny and the rest are byte-identical
between 4.12 and 5.0 when handed identical input. Everything below is that one step's
difference, amplified.

### What it becomes by the end of the pipeline

Whole pipeline in each environment. Per-frame worst pixel, and the share of the frame that
changed at all:

| stage | max abs difference (median / p95 / worst frame) | pixels changed (median / worst frame) |
|---|---|---|
| `roi` | 4 / 5 / **7** grey levels | 4.0 % / 41.4 % |
| `gray` | 4 / 5 / 7 | 5.3 % / 55.3 % |
| `clahe` | **8 / 11 / 12** | 15.2 % / 66.8 % |
| `blur` | 2 / 4 / 4 | 15.2 % / 44.0 % |
| `edges` | 255 / 255 / 255 *(binary — a pixel is on or off)* | 0.04 % / 2.0 % |
| `resized` | 32 / 112 / 128 | 0.35 % / 19.7 % |
| `tensor` | 0.126 / 0.439 / **0.502** of the [0, 1] range | 0.35 % / 19.7 % |

Read the `clahe` row against the `roi` row above it: CLAHE roughly **doubles** the
difference it is handed — a median worst pixel of 4 grey levels going in, 8 coming out,
12 in the worst frame. That is what CLAHE is for. It is a local contrast stretch, so it
amplifies whatever local difference exists, including one that came from a warp.

At `edges`, up to **5 298 of 262 144** interior pixels flip between on and off in a single
frame. **Every one of the 120 frames differs at the final tensor.**

### What it does to the quality gate

Of the 120 frames, 94 reach the three measured rules (the other 7 fail on marker count,
before anything is measured, and the rest are deliberately spoiled).

| gate metric | median shift | worst shift | closest any accepted frame comes to its threshold | worst shift as a share of that margin |
|---|---|---|---|---|
| `laplacian_variance` | 1.43 | 2.00 | 126.16 | 1.58 % |
| `mean_luminance` | 0.0038 | 0.0055 | 60.37 | 0.01 % |
| `edge_density` (interior) | 1.2 × 10⁻⁵ | 7.0 × 10⁻⁵ | 0.00747 | 0.94 % |

**No frame changed verdict**, and at commit `a4639f7` re-ingesting all four synthetic
sessions under the pin left `data/metadata.csv` byte-identical. The gate has margin; the
pixels do not.

## Decision

`config/versions.json` pins **`opencv-contrib-python` 4.12.0**. `python tasks.py check-versions`
asserts the distribution, the absence of a conflicting one, that `cv2.aruco` answers, and the
minor version — in that order, because on a machine with plain `opencv-python` the version
string is *right* and everything else is wrong.

Moving off this baseline is a **dataset migration**, not a dependency bump. It requires
re-ingesting every capture and retraining, and an ADR that supersedes this one.

## Why the measurement changes the reason

The original justification was that aligning was free because nothing existed yet. That is
true and it is not the point. The synthetic captures are rendered with generous, even
contrast; a real low-contrast classroom capture is the case §5.3a's warning is about, and on
it those grey levels move **contour pixels**, not just numbers — the difference survives
into the edge map, which is the entire input the model sees.

The final tensor is not saved by the 50/150 Canny thresholds either. On a single
well-behaved image it can be — the thresholds absorb a small shift — but across 120 frames
every single tensor differs. Anyone reasoning from one image would conclude the pipeline is
robust to the version. It is not.

So the pin is not tidiness. Two OpenCV versions produce two different datasets from the same
photographs, a model trained on one does not match the other, and nothing in the pipeline
would say so: `pytest` passes under both, because the tests assert shapes, dtypes, ranges and
determinism — properties that hold in either version.

## Consequences

- `docs/03-preprocessing-spec.md` §5.3a's "resolve the pin before the first real ingest" now
  has its cost attached. Before F3b it is free; after it, it is 4 000 photographs re-ingested.
- The version belongs in the TDR methodology alongside the seed and the card, and this table
  is the evidence for the sentence that says so.
- `pytest` is not a guard against this and was never going to be. `check-versions` is.
- The result is reproducible: `tools/opencv_baseline_diff.py` regenerates every number in this
  file from two interpreters and the synthetic sessions.

## Alternatives rejected

**Move the pin to 5.0.0.** Defensible on the day it was raised — no data existed. Rejected
because 4.12 is what the whole repository, `requirements.txt` and the F0 environment already
specify, and because the pin's job is to be stable, not current. Chasing the newest OpenCV
means re-ingesting on someone else's release schedule.

**Pin only the major version.** The measurement shows the divergence is at a major boundary
(4 → 5), so this is not obviously wrong. Rejected anyway: nothing establishes that 4.12 and a
hypothetical 4.20 agree, and the check that would find out is the one being weakened.

**Accept the difference and re-ingest when convenient.** This is the option §5.3a exists to
forbid. The cost is not the re-ingest; it is the window in which some processed images came
from one version and some from another, with nothing in `metadata.csv` recording which.

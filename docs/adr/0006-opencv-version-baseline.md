# ADR 0006 — The OpenCV version is part of the experimental method

**Status:** accepted, 2026-08-25
**Required by** `docs/03-preprocessing-spec.md` §5.3a, which says a resolved version pin must be
recorded in an ADR with the reason. The reason turned out not to be the one anybody expected.

## Context

`config/versions.json` pinned `opencv-contrib-python` 4.12.0 while the working environment had
5.0.0 on Python 3.14. §5.3a offers two resolutions — move the pin to the environment, or the
environment to the pin — and forbids the third, which is leaving `check-versions` red.

The environment was moved to the pin, and the argument at the time was hygiene: no dataset and
no checkpoint existed, so aligning cost nothing.

Then it was measured over 120 frames, and the measurement says something different.

## The measurement

`tools/opencv_baseline_diff.py`, over the 120 frames of `data/raw/SYNTH-20260101-01`,
`config/preprocess.json` at 1.1.0. Both environments on **Python 3.11.9 and numpy 2.1.3**, so
the only variable is OpenCV. (Running the 5.0.0 side on Python 3.14.4 / numpy 2.5.2 gives
byte-identical output to the 3.11 run, which is what licenses attributing everything below to
OpenCV alone.)

| | pinned | compared against |
|---|---|---|
| distribution | `opencv-contrib-python` 4.12.0.88 | `opencv-contrib-python` 5.0.0.93 |
| `cv2.__version__` | 4.12.0 | 5.0.0 |

### One operation is responsible: `cv2.warpPerspective`

Each stage fed the **pinned** environment's output of the previous stage, so every row is that
operation's own contribution and not the sum of everything upstream:

| step | stage | max abs difference | frames differing |
|---|---|---|---|
| 2 | `roi` — `warpPerspective`, `INTER_LINEAR` | **7 grey levels** | **120 / 120** |
| 3 | `gray` — `cvtColor` | 0 | 0 / 120 |
| 4 | `clahe` — `createCLAHE().apply` | 0 | 0 / 120 |
| 5 | `blur` — `GaussianBlur` | 0 | 0 / 120 |
| 6 | `edges` — `Canny` | 0 | 0 / 120 |
| 7 | `resized` — `resize`, `INTER_AREA` | 0 | 0 / 120 |
| 8 | `tensor` — scale | 0 | 0 / 120 |

CLAHE, Canny, the blur and the resize are **byte-identical** between 4.12 and 5.0 when handed
identical input. The divergence enters at step 2 and everything after it is that divergence
carried forward.

### CLAHE is the amplifier, not the origin

Whole pipeline in each environment. Per-frame worst pixel, and the share of the frame that
changed at all:

| stage | max abs difference (median / p95 / worst frame) | pixels changed (median / worst frame) |
|---|---|---|
| `roi` | **4** / 5 / 7 grey levels | 4.0 % / 41.4 % |
| `gray` | 4 / 5 / 7 | 5.3 % / 55.3 % |
| `clahe` | **8** / 11 / **12** | 15.2 % / 66.8 % |
| `blur` | 2 / 4 / 4 | 15.2 % / 44.0 % |
| `edges` | 255 / 255 / 255 *(binary — a pixel is on or off)* | 0.04 % / 2.0 % |
| `resized` | 32 / 112 / 128 | 0.35 % / 19.7 % |
| `tensor` | 0.126 / 0.439 / **0.502** of the [0, 1] range | 0.35 % / 19.7 % |

Read the `clahe` row against the `roi` row above it. CLAHE **doubles** what it is handed —
median worst pixel 4 grey levels going in, 8 coming out, 12 in the worst frame — while
contributing nothing of its own. That is what a local contrast stretch does: it amplifies
whatever local difference exists, including one that arrived from a warp two steps earlier.

At `edges`, up to **5 298 of 262 144** interior pixels flip between on and off in a single
frame. **All 120 tensors differ.**

### What it does to the quality gate

Of the 120 frames, 94 reach the three measured rules (7 fail on marker count before anything is
measured; the rest are deliberately spoiled).

| gate metric | median shift | worst shift | closest any accepted frame comes to its threshold | worst shift as a share of that margin |
|---|---|---|---|---|
| `laplacian_variance` | 1.43 | 2.00 | 126.16 | 1.58 % |
| `mean_luminance` | 0.0038 | 0.0055 | 60.37 | 0.01 % |
| `edge_density` (interior) | 1.2 × 10⁻⁵ | 7.0 × 10⁻⁵ | 0.00747 | 0.94 % |

**No frame changed verdict**, and at commit `a4639f7` re-ingesting all four synthetic sessions
under the pin left `data/metadata.csv` byte-identical.

**The gate has margin; the pixels do not.** Those two sentences describe the same measurement
and lead to opposite conclusions, which is why both are here. A reader who checks only the
verdicts concludes the version does not matter. It matters at every pixel the model reads.

## Decision

`config/versions.json` pins **`opencv-contrib-python` 4.12.0**, and the OpenCV version is
recorded as part of the **experimental method** — beside the seed, the camera distance and the
name of the GPU — not as a property of a developer's machine.

Concretely:

1. `python tasks.py check-versions` asserts the distribution, the absence of a conflicting one,
   that `cv2.aruco` answers, and the minor version — in that order, because on a machine with
   plain `opencv-python` the version string is *right* and everything else is wrong.
2. Every checkpoint records `cv2.__version__` and `opencv_package`, and
   `python tasks.py check-model` refuses a mismatch as hard as it refuses a preprocessing hash
   mismatch (`docs/03-preprocessing-spec.md` §4). The config hash alone does not cover this: a
   reinstalled venv leaves the hash matching and the pixels changed.
3. The TDR methodology names the version. A reader reproducing this work needs it for the same
   reason they need `d = 210 mm`.

Moving off this baseline is a **dataset migration**, not a dependency bump: re-ingest every
capture, retrain, and supersede this ADR.

## Why this is not the reason the pin was originally argued for

The first justification was that aligning was free because nothing existed yet. True, and beside
the point.

The claim that went with it — that the final tensor survives because Canny's 50/150 thresholds
absorb the difference — came from **one image**, and it is false. Across 120 frames every tensor
differs. A single-sample measurement cannot support a statement about a pipeline, and this one
happened to support the comforting version: anyone reasoning from that image would conclude the
pipeline is robust to the OpenCV version.

Nor is CLAHE the threat, which is what §5.3 and §5.3a used to imply and no longer do. CLAHE is
bit-exact across the two versions. It is dangerous only in the sense that it faithfully doubles
an error handed to it — which makes the geometric step upstream, not the contrast stretch, the
thing to pin and the thing to watch.

Two OpenCV versions produce two different datasets from the same photographs, a model trained on
one does not match the other, and nothing in the pipeline says so: `pytest` passes under both,
because the tests assert shapes, dtypes, ranges and determinism — properties that hold in either
version.

## Consequences

- §5.3a's "resolve the pin before the first real ingest" now has its cost attached. Before F3b it
  is free; after it, it is 4 000 photographs re-ingested and every model retrained.
- The version goes in the TDR methodology section, not in a footnote about the environment.
- `pytest` is not a guard against this and never could be. `check-versions` and `check-model` are.
- Every figure in this file regenerates from two interpreters and the synthetic sessions:

  ```
  <pinned-python>  tools/opencv_baseline_diff.py dump  out/pinned --mode chain
  <other-python>   tools/opencv_baseline_diff.py dump  out/other  --mode chain
  <any-python>     tools/opencv_baseline_diff.py compare out/pinned out/other
  <other-python>   tools/opencv_baseline_diff.py dump  out/iso --mode isolated --reference out/pinned
  ```

## Alternatives rejected

**Move the pin to 5.0.0.** Defensible on the day it was raised — no data existed. Rejected
because 4.12 is what the repository, `requirements.txt` and the F0 environment already specify,
and because a pin's job is to be stable rather than current. Chasing the newest OpenCV means
re-ingesting on someone else's release schedule.

**Pin only the major version.** The divergence measured here is at a major boundary (4 → 5), so
this is not obviously wrong. Rejected anyway: nothing establishes that 4.12 and a hypothetical
4.20 agree on `warpPerspective`, and the check that would find out is the one being weakened.

**Accept the difference and re-ingest when convenient.** The option §5.3a exists to forbid. The
cost is not the re-ingest; it is the window in which some processed images came from one version
and some from another, with nothing in `metadata.csv` recording which.

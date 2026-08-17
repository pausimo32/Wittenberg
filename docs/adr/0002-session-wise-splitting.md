# ADR 0002 — Split the dataset by capture session, not randomly

- **Date:** 2026-08-16
- **Status:** accepted

## Context

Data is collected in bursts: the rig is fixed, the frequency is set, and dozens of frames are
captured seconds apart. Consecutive frames of the same class are near-duplicates — same
lighting, same sand grains, same camera pose.

Under a random train/test split, near-duplicates of a test image sit in the training set. The
model can score highly by recognising the session rather than the nodal geometry. The
resulting accuracy is not an estimate of performance on a new photo.

The reference study (Kumar & Wani 2024) reports 200 training and 5 test images per class and
does not describe how they were separated. Its ≥ 80 % figures may or may not be affected; we
cannot tell, and neither can a reader.

## Decision

Split by `session_id` using `GroupShuffleSplit`. A session appears in exactly one of
train / val / test. Minimum 4 sessions; leave-one-session-out cross-validation.

Report **both** the naive random-split accuracy and the session-wise accuracy in the TDR.

## Consequences

- Headline accuracy will be lower than a random split would give, and lower than the reference
  study's. This is correct, and the gap is a finding worth a paragraph in the discussion.
- Data collection cost rises: ≥ 4 sessions, each covering every class. Cannot be shortcut by
  taking more frames in one sitting.
- Confidence intervals are computed over independent captures, not frames
  (`docs/02-dataset-spec.md` §5).
- `data/metadata.csv` must carry `session_id` for every image, with no exceptions — an image
  with an unknown session cannot be used.

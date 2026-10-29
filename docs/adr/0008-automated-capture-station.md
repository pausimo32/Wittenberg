# ADR 0008 — An automated capture station: MacBook, iPhone 13, one page of the site

- **Date:** 2026-10-22
- **Status:** **proposed.** Accepted once the feasibility probe below has run on the real
  hardware. Every figure marked *computed* is derived from published specifications and the
  rig geometry; none has been measured yet. Replace each with its measurement, and say so.
- **Amends** `docs/02-dataset-spec.md` §1 and §4 (camera, frame format, framing) and
  `docs/05-webapp-spec.md` §4 (one more page). ADR 0004, 0005 and 0007 are untouched: the
  station is a page of the same local app, served by the same FastAPI process on `127.0.0.1`.
- **Trigger:** collecting ≥ 4 sessions × ≥ 60 frames × every class by hand — set the generator,
  wait, press the shutter, write down the frequency — is hours of error-prone repetition per
  session. The team asked for a tool that drives the speaker and the camera itself.

## Context

Three facts shaped the decision.

1. **The storage half already exists.** `src/dataset_api.py` implements the whole `/dades`
   flow of `docs/05` §9 on the server: draft session with a validated `session.json`, ArUco
   calibration that must come first, per-frame upload gated by `ingest.assess_frame` — the
   ingest gate itself, not a copy — SHA-256 de-duplication, and a commit that ends in
   `data/raw/` and `metadata.csv`. What is missing is a client in `site/`; the only one is the
   React app in `web/`, which ADR 0007 is retiring.
2. **The phone is an iPhone 13, not the iPhone 11** `docs/02` §1 was written for. No real
   capture exists yet — every session in the repo is `SYNTH-*` — so changing the camera now
   costs no data.
3. **An iPhone's native shutter cannot be triggered from a computer reliably.** On a Mac it does
   not need to be: Continuity Camera exposes the iPhone as an ordinary camera to the browser.

## Decision

**1. The station is a page of the site, `site/acquisition.html`, and a client of the existing
`/api/drafts` endpoints.** No new storage path, no second gate. A frame the station accepts is a
frame `python tasks.py ingest` accepts, because it is the same call (`docs/05` §9, "the gate is
the same gate").

**2. Hardware: a MacBook with the iPhone 13 as a Continuity Camera.** The page reads it with
`getUserMedia`, as `analyseur.html` already reads a camera. Nothing in the page is specific to
macOS: on another computer the same page works with any camera the browser can see. The app's
portability (ADR 0005) is unchanged; the Mac is where the station happens to run.

**3. The tone is generated in the page with Web Audio** — a sine `OscillatorNode` at the class
frequency, with gain ramps at start and stop, out of the Mac's audio output into the amplifier
and the plate's driver. No new dependency, no native audio library.

**4. Frames are stored as PNG, grabbed from the video stream.** Lossless, so no second
compression is added to the one the stream already applied (`docs/03` §5.2). `src/imageio.py`
already decodes PNG. The session log records where the frames came from (decision 7).

**5. Capture order is round-robin, not per class.** One *round* visits every class once, in an
order shuffled per round from a seed recorded in the session log. A session is ≥ 60 rounds.
Capturing sixty frames of one class and then sixty of the next would make anything that drifts
during a session — sand lost over the edges, the amplifier warming, the light changing — a
function of the class label, which is the condition-confound trap of `docs/00-master-plan.md`
F3b reappearing *inside* a session. Round-robin also means every frame of a class is a pattern
that re-formed after the plate played a different mode, rather than a near-duplicate of the
frame one second earlier.

**6. A frame is captured only after these checks, in this order:**

| # | Check | Where | On failure |
|---|---|---|---|
| 1 | dwell elapsed (fixed, `dwell_s` of the session log) | page | wait |
| 2 | **pattern settled** — two consecutive stream frames differ by less than a threshold | page | keep waiting, up to a timeout, then skip the class this round and log it |
| 3 | **markers where the calibration put them** — the four ArUco corners within a few px of the session's calibration | server | reject: the phone or the plate moved. Pause the session; the calibration no longer describes the frame |
| 4 | **the ingest gate** — markers, blur, luminance, interior edge density (`docs/02` §6) | server, existing | retry up to N times, then log it and move on |
| 5 | *optional:* **the tone is the tone** — FFT of the Mac's microphone, peak within tolerance of the class frequency | page | pause: the speaker is not playing what the label says |

Check 3 is new, and it is the check a person would never make by hand: Continuity Camera picks
its orientation from the phone's attitude, and a zenithal phone is the attitude it is least sure
about. A frame that rotated after the calibration warps to the wrong quadrilateral and still
looks like a plausible plate.

**What is deliberately not a check: the classifier.** Accepting only frames the model
recognises would select the training set by the model's current opinion — the dataset would
grow towards what is already easy, and the reported accuracy would be measured on data filtered
by its own predictor.

**7. `session.json` gains an optional `camera` key**, e.g. `"iphone13-continuity-png"` for the
station and `"iphone13-heic"` for photos taken by hand. Optional, because ingest refuses only
on *missing required* keys and copies the whole log into `data/ingest_report.json`; no code
changes to carry it. It is what lets the TDR say which frames came from where.

**8. The camera is the same physical iPhone 13 in every session.** Like lighting and material
(`docs/00-master-plan.md` F3b), a camera used in only one session predicts the session-wise split
perfectly, and the test accuracy would then measure "a different phone".

## Consequences

**`docs/02` §4's "untouched, straight off the phone" changes meaning.** A station frame is a
video frame, compressed by the stream before the browser sees it, and at stream resolution
(probably 1920 px wide) rather than 4032 px. It is stored byte for byte as it left the page,
which keeps §9's raw-bytes rule, but it is not a HEIC photograph. The model is then trained on
stream frames and used, in `analyseur.html`, on photographs. Whether that gap costs accuracy is
an empirical question with a ready-made test: photograph a few frames per class by hand in one
session (`camera: "iphone13-heic"`) and evaluate on them separately. Report the result in the
TDR either way.

**Resolution is not the constraint.** The ROI is 512 px over a 180 mm plate, so it needs
≥ 2.84 px/mm in the source to avoid upsampling. A 1920 px stream across the frame width gives
that up to a working distance of ≈ 500 mm (*computed*: iPhone 13 wide camera, 5.1 mm focal
length, 6.85 mm sensor width).

**Framing is the constraint, and it already was** — see `docs/02` §1.2. In short: a 16:9 stream
is ≈ 25 % shorter than the 4:3 photo the 210 mm distance was computed for, and the markers sit
*outside* the plate. The working distance has to be re-derived from the real marker size and
the real stream format, which is the first thing the probe measures.

**Exposure may not be lockable.** `docs/02` §1 requires AE/AF locked. Whether Continuity Camera
honours exposure and focus constraints from the browser is unknown. If it does not, the station
records the mean ROI luminance of every frame and flags drift from the calibration frame; the
gate's luminance range is the backstop it already was.

**macOS video effects must be off.** Center Stage re-frames the image by itself; Portrait,
Studio Light and Reactions alter it. Any of them breaks the calibration or the pixels. The page
reminds the operator before the calibration step; it cannot switch them off.

**The tab must stay in front, and the Mac awake.** Browsers throttle timers in background tabs,
and a session is about an hour. The page holds a screen Wake Lock and says so if it loses it.

**Sand needs replenishing.** The page pauses every K rounds (K in the session log) for the
operator to re-seed, and records each pause.

**Time.** About 7 s per frame (*computed*: dwell + settle + upload + gate), so ten classes × 60
rounds ≈ 70 minutes, unattended between re-seeding pauses.

## The feasibility probe — what turns this ADR from proposed to accepted

A throwaway page, outside `site/`, run once on the MacBook with the iPhone 13 attached:

1. the camera list and the stream's maximum resolution and aspect ratio;
2. whether `exposureMode`/`focusMode` constraints are supported, and honoured;
3. one calibration frame through `calibrate`, one pattern frame through `ingest.assess_frame`;
4. the framing: plate plus markers, at which distance, on the actual mount;
5. a test tone through the actual amplifier, and the microphone's FFT of it.

Each answer replaces a *computed* or *unknown* above.

## Rejected alternatives

**The phone opens the page over Wi-Fi.** The server would have to listen beyond `127.0.0.1`,
which ADR 0004 forbids, and browsers grant camera access only on secure origins, so it would
also need HTTPS certificates on a laptop. Two rules broken to avoid a cable.

**A virtual-webcam app (Camo, Iriun, DroidCam).** Works, and is what a Windows station would
use. On a Mac, Continuity Camera does the same with nothing to install.

**Native HEIC photos, shutter triggered from the computer.** No reliable mechanism exists for an
iPhone. ADB does this for Android, which the team does not use.

**Generating the tone in Python** (`sounddevice`/PortAudio). A native dependency on every machine,
for what one `OscillatorNode` does in the page that already owns the timing.

**Per-class bursts, as `docs/00` F3b step 3 literally reads.** See decision 5.

## What to say in the TDR

That the captures were automated, and how: tone, dwell, settle check, the ingest gate, the
round-robin order, and why that order matters. Report the station's yield by rejection reason
exactly as for any session (`docs/06-research-protocol.md`). State that training frames are video
frames and test the gap to photographs rather than assume it away.

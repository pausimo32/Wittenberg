# Research protocol — what the TDR needs

Read this when producing anything that ends up in the written work. It converts the Pla de
treball into obligations on the code.

## 1. Research questions → artefacts

| # | Question (Pla de treball §1) | Artefact that answers it |
|---|---|---|
| P1 | Can a system identify the frequency from a photo with our own CNN? | `runs/<ts>/metrics.json`, confusion matrix, `/analitza` working |
| P2 | Does the tool make wave concepts intuitive for non-physicists? | pre/post-test deltas per concept |
| S3 | Which age/education group improves most? | pre/post deltas segmented by group |
| S4 | Which experimental conditions give the sharpest patterns? | ablation A8 + `data/ingest_report.json` yield by condition |
| S5 | How accurate is supervised classification by frequency? | per-class F1 + Wilson interval (`docs/04-model-spec.md` §4) |
| S6 | What are the main error sources and limits? | frequency-distance error analysis, Grad-CAM, `docs/01-domain-physics.md` §3 |

If a change does not serve one of these rows, it is out of scope.

## 2. Annex generation — automated, never manual

The Pla de treball lists 12 annexes. These are produced by `python tasks.py annexes`, from data, into
`annexes/`. Regenerating must be idempotent.

| Annex | Source |
|---|---|
| 3. Frequency table | `data/modes.csv` |
| 4. Sample of obtained patterns | `data/metadata.csv` + `data/processed/` |
| 6. Image database | `data/metadata.csv` summary tables |
| 7. Preprocessing code and stages | `src/preprocess.py` syntax-highlighted, plus the 8-stage strip rendered for one image per class |
| 8. Model code | `src/model.py`, `src/train.py` |
| 9. Full model results | `runs/<ts>/report/` |
| 10. App screenshots | Playwright against the built local app, fixed viewport — include the 13-stage viewer and the /dades summary matrix |

Annexes 1, 2, 5, 11, 12 (schematics, photos, protocol, questionnaires) are produced by hand.

A number that appears in the TDR body must be traceable to a file in `runs/`. No figure is
retyped from memory.

## 3. Field-study constraints on the software

- The `/apren` module must be **frozen** before the first session. A change mid-study makes
  groups non-comparable. Tag the release (`v1.0-field`) and record the tag in every session log.
- The pre-test is administered before any exposure to the app, the post-test immediately
  after, with equivalent — not identical — questions (Pla de treball §3.2).
- Participants are not told a post-test is coming (Pla de treball §3.2). The app must not
  hint at it either.
- Offline-capable, because a Wi-Fi failure loses a whole session.
- Informed consent, and for minors, guardian consent. No personal data in the app, ever.
  Questionnaire identifiers are random codes so pre and post can be paired without names.

## 4. Reproducibility claims the TDR will make

Every claim below must be true at submission:

1. `git clone` + `python tasks.py setup` + `python tasks.py app` gives a working tool on any
   computer, with no GPU and no internet. Re-training reproduces reported metrics within ±1 %
   (seed 42) on the same GPU, and the card is named in the methodology.
2. Every reported figure regenerates from `runs/` with one command.
3. The preprocessing used at inference **is** the one used at training — the API calls
   `src/preprocess.py`, the same function that built the dataset, and refuses to start if the
   checkpoint's config hash differs (`docs/adr/0004`). State this as an architectural property,
   not as a test result.
4. The dataset split is by session and stated as such, with the naive-split number reported
   alongside for comparison.
5. Every reported acceptance yield carries the gate that produced it. `python tasks.py ingest`
   copies `config/quality_gate.json` verbatim into `data/ingest_report.json`, so the thresholds
   travel with the figure instead of having to be read out of whichever commit is checked out.

### Yield is a reported number, so it has reported inputs

Research question S4 is answered from `data/ingest_report.json`. A sentence like "27.5 % of
frames were rejected" is not reproducible on its own: it depends entirely on four thresholds.
The report therefore carries, per session:

- `gate` — `config/quality_gate.json`, verbatim, as applied to that ingest
- `session_log` — the whole of `data/raw/<session_id>/session.json`, including `operator` and
  `plate`, which are not `metadata.csv` columns and live nowhere else that is committed
- `rejected` — the count by reason, and every rejected frame by path with its measurements
- `flags` — anything the calibration reported about the rig, such as a marker taped on rotated
  (`docs/02-dataset-spec.md` §1.1)

Quote the rejection rate with its thresholds, or do not quote it.

## 5. Writing

- Language: Catalan. Register: formal, first person plural.
- Structure: fixed by the Pla de treball §6 index. Do not reorganise it.
- Citations: APA 7, from the Mendeley library. Never invent a reference or a DOI.
- When reporting the reference study, cite it as Kumar & Wani (2024) and state clearly which
  of our choices replicate it and which deliberately deviate (CLAHE, session-wise splitting,
  GAP head, larger test set) — a documented deviation is a contribution; an undocumented one
  is an error.
- Limitations are a section, not an apology. Small sample, one plate, one camera, one room.

---
name: tdr-annex-export
description: Generate the TDR annexes (frequency table, pattern samples, database summary, code listings, full model results, app screenshots) from data and runs. Use when preparing a TDR draft or after a training run that will be reported.
allowed-tools: Read Bash Glob
---

# Export TDR annexes

Contract: `docs/06-research-protocol.md` §2. Annexes are **generated**, never assembled by hand.
Idempotent: running twice produces identical files.

```bash
make annexes RUN=runs/20260920-1432
```

## What it produces

| Annex | File | Source |
|---|---|---|
| 3 | `annexes/A3_taula_frequencies.pdf` | `data/modes.csv` |
| 4 | `annexes/A4_mostra_patrons.pdf` | `data/metadata.csv` + `data/processed/` |
| 6 | `annexes/A6_base_de_dades.pdf` | `metadata.csv` summary tables |
| 7 | `annexes/A7_codi_preprocessament.pdf` | `src/preprocess.py`, `web/lib/preprocess.ts` |
| 8 | `annexes/A8_codi_model.pdf` | `src/model.py`, `src/train.py` |
| 9 | `annexes/A9_resultats_model.pdf` | `<RUN>/report/` |
| 10 | `annexes/A10_captures_webapp.pdf` | Playwright, fixed viewport |

Annexes 1, 2, 5, 11, 12 are manual (schematics, rig photos, protocol, both questionnaires).
The Makefile lists them as `MANUAL` and does not fabricate them.

## Rules

1. **Every number in the TDR body traces to a file under `runs/`.** If `make annexes` cannot
   produce a figure, the figure does not go in the document.
2. Captions and headers in Catalan, via `docs/07-glossary-ca-en.md`. `exactitud` = accuracy,
   `precisió` = precision.
3. Screenshots come from the tagged field release (`v1.0-field`), not from `main` — the
   document must show what participants actually saw.
4. Annex 9 includes both the session-wise and the naive-split numbers, labelled, with the gap
   stated (`docs/adr/0002`).
5. Record the run ID and git SHA on the first page of every generated annex. A reviewer must be
   able to regenerate it.

## Check before submitting

- [ ] Every `[PENDENT: ...]` marker in the draft resolved
- [ ] Every figure referenced in the body exists in `annexes/`
- [ ] Annex numbering matches the Pla de treball §6 index exactly
- [ ] `runs/<id>` referenced in the annexes still exists in the repo

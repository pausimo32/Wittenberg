# Web app specification

Next.js App Router, TypeScript, Tailwind, **static export** (`output: 'export'`). No server,
no database, no image upload leaves the device. Deployable to GitHub Pages / Netlify free tier.

Privacy is not a nice-to-have here: participants are minors and photos are taken in schools.
"Nothing leaves your phone" is a sentence that must be literally true and stated in the UI.

## 1. Routes

| Route | Purpose |
|---|---|
| `/` | landing — one sentence, one button, one example |
| `/analitza` | **Analyse.** Camera or file → prediction |
| `/apren` | **Learn.** The 7-concept didactic sequence (`docs/01-domain-physics.md` §4) |
| `/galeria` | **Gallery.** The project's own modes, ordered by frequency |
| `/sobre` | **About.** Method, limits, team, TDR reference, licence |

Catalan slugs. UI copy Catalan, code English (`CLAUDE.md`).

## 2. `/analitza` — the pipeline in the browser

```
<input capture> or drop
  → createImageBitmap(blob, {imageOrientation:'from-image'})
  → lib/preprocess.ts   (opencv.js — EXACTLY docs/03-preprocessing-spec.md)
  → onnxruntime-web (WASM, SIMD + threads where available)
  → softmax → top-3
```

Rules:

- `opencv.js` and the `.onnx` are **lazy-loaded** on first use, behind a progress bar. They
  must not be in the initial bundle — that would make the landing page a 12 MB download.
- The app **asserts** `bundle.preprocess_version === config.preprocess_version` at load and
  refuses to predict on mismatch (`docs/04-model-spec.md` §7).
- Everything runs in a Web Worker. The main thread never blocks.
- No network request after load. Verify with a Playwright test that asserts zero outbound
  requests during a prediction.

### Result presentation — the honesty requirement

Show top-3 with probabilities, never a bare single number.

| Max probability | Behaviour |
|---|---|
| ≥ 0.70 | "Probablement **N Hz**" + the other two candidates |
| 0.40 – 0.70 | "Podria ser **N Hz** o **M Hz**" — both shown equally |
| < 0.40 | "No ho sé prou" + guidance: more sand, wait longer, check the frequency is on resonance |

Also show, always: the preprocessed 128×128 edge image next to the original. Seeing what the
model actually sees is half the didactic value, and it makes failures self-explanatory.

Never round a prediction to a "nicer" number, and never hide low confidence.

## 3. `/apren` — the didactic module

Drives the field research. Its design is constrained by `docs/06-research-protocol.md`.

- Seven steps, in the order of `docs/01-domain-physics.md` §4. Linear, no skipping ahead.
- Each step: one sentence, one interactive element, one check-question. No walls of text.
- **WebAudio tone generator** — sine, 100–5000 Hz, with a prominent volume control and a
  −20 dB default. Warn before playing. Ship a mute-first state.
- **Sweep animation** — pre-rendered from the project's own captures, not simulated. Showing
  real photos of a real plate is the point.
- Works offline after first load (service worker) — school Wi-Fi is not to be trusted.
- **No analytics, no cookies, no localStorage of anything personal.** Session progress in
  memory only. Questionnaire responses go through Google Forms, separately, as planned.
- Accessibility: keyboard-navigable, `prefers-reduced-motion` respected, all interactive
  elements ≥ 44 px, contrast ≥ 4.5:1. Participants include older adults.

## 4. `/galeria`

Grid of the project's modes ordered by frequency, each with: photo, edge map, frequency,
number of nodal lines. Doubles as TDR annex 4. Generated from `data/metadata.csv` at build
time — never hand-maintained.

## 5. Performance budget

| Metric | Budget |
|---|---|
| Initial JS (landing) | ≤ 150 kB gzipped |
| Lazy chunk (opencv.js + ort + model) | ≤ 12 MB, cached |
| Time to first prediction, mid-range Android, after load | ≤ 1.5 s |
| Lighthouse performance / accessibility | ≥ 90 / ≥ 95 |

## 6. Testing

- Unit: `lib/preprocess.ts` against fixtures from `tests/golden/`
- Parity: the browser half of `npm run parity` (`docs/03-preprocessing-spec.md` §4)
- E2E (Playwright): upload → prediction; offline mode; zero-network assertion
- Visual regression on `/apren` steps, since a broken didactic step invalidates a session of
  field data that cannot be recollected

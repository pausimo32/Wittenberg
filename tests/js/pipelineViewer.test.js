/**
 * The pipeline viewer's rules (`docs/05` §8), separated from the DOM that obeys them.
 *
 * Everything asserted here is a rule the spec states in prose: label by key and never by
 * the API's own title, show a measured time or nothing at all, print a config value as it
 * is rather than as it reads nicely, linger on Canny, pause when the user takes over.
 * The DOM half of `pipelineViewer.js` contains no rule of its own, so this file is the
 * whole of the viewer's logic.
 *
 * Run with `node --test "tests/js/*.test.js"` — the bare `tests/js/` form stopped working
 * at Node 24, which reads it as a module specifier rather than a directory.
 */
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const viewer = require('../../site/js/pipelineViewer.js');

/**
 * `site/js/i18n.js` is a browser IIFE that hangs `WI18n` on `window`, so it is run here in
 * a sandbox with the four browser objects it touches rather than being restructured into a
 * module for the sake of a test. The alternative — splitting the 1350-line translation
 * table into its own requireable file — is a large edit to buy what twenty lines buy here.
 */
function loadI18n() {
  const store = {};
  const sandbox = {
    window: {},
    localStorage: {
      getItem: (k) => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
    },
    document: {
      documentElement: {},
      querySelectorAll: () => [],
      dispatchEvent: () => {},
    },
    CustomEvent: function (type, init) { this.type = type; this.detail = init && init.detail; },
  };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, '../../site/js/i18n.js'), 'utf8'),
    sandbox
  );
  return sandbox.window.WI18n;
}

/** The thirteen keys `src/api.py` sends, in order. §8 fixes the list. */
const STAGE_KEYS = [
  'capture', 'roi', 'gray', 'clahe', 'blur', 'edges', 'resized', 'tensor',
  'features_1', 'features_2', 'features_3', 'gradcam', 'probabilities',
];

/** Stage 6 as `/api/predict` really sends it — params straight from config/preprocess.json. */
const CANNY = {
  n: 6,
  key: 'edges',
  title: 'Canny',
  kind: 'image',
  call: 'cv2.Canny',
  params: { threshold1: 50, threshold2: 150, aperture_size: 3, l2_gradient: false },
  shape: [512, 512],
  dtype: 'uint8',
  display_shape: null,
  ms: 0.353,
  png_base64: 'iVBORw0KGgo=',
};

const CAPTURE = {
  n: 1,
  key: 'capture',
  title: 'Captura',
  kind: 'image',
  call: 'imageio.decode_bytes',
  params: {},
  shape: [3024, 4032, 4],
  dtype: 'uint8',
  display_shape: [576, 768, 4],
  ms: 18.9,
  png_base64: 'iVBORw0KGgo=',
};

const PROBABILITIES = {
  n: 13,
  key: 'probabilities',
  title: 'Probabilitats',
  kind: 'bars',
  call: 'torch.softmax',
  params: { dim: 1 },
  ms: null,
  bars: [
    { class_index: 0, freq_hz: 420, p: 0.02 },
    { class_index: 2, freq_hz: 860, p: 0.97 },
  ],
};

// ── Labelling ───────────────────────────────────────────────────────────────

test('a stage is labelled by its key, never by the title the API sends', () => {
  // The API answers in Catalan (`Escala de grisos`); the UI is French by default with a
  // ca/es/fr switcher (ADR 0007). Reading `stage.title` would pin one language onto all
  // three and make the switcher lie on this one screen.
  assert.strictEqual(viewer.labelKey({ key: 'gray', title: 'Escala de grisos' }), 'viewer.stage.gray');
  assert.strictEqual(viewer.captionKey({ key: 'gray', title: 'Escala de grisos' }), 'viewer.caption.gray');
});

test('every stage the API can send has a French label and caption', () => {
  // §8 fixes the thirteen stages and `src/api.py` fixes their keys. `WI18n.t` returns the
  // key itself when it knows no string, so a stage nobody translated renders the literal
  // `viewer.stage.gradcam` on screen. French is the default language (ADR 0007) and the
  // fallback for the other two, so this is the check that has to be strict.
  const i18n = loadI18n();
  for (const key of STAGE_KEYS) {
    for (const prefix of ['viewer.stage.', 'viewer.caption.']) {
      assert.notStrictEqual(i18n.t(prefix + key), prefix + key, `no French string for ${prefix}${key}`);
    }
  }
});

test('the captions are really translated, not left sitting in French', () => {
  // `WI18n.t` falls back to French, so a missing Catalan string is invisible to the check
  // above. The captions are whole sentences: two languages never write one the same way by
  // accident, which the labels — `CLAHE`, `Canny`, `Grad-CAM` — genuinely do.
  const i18n = loadI18n();
  const french = STAGE_KEYS.map((key) => i18n.t('viewer.caption.' + key));
  for (const lang of ['es', 'ca']) {
    i18n.setLang(lang);
    STAGE_KEYS.forEach((key, i) => {
      assert.notStrictEqual(
        i18n.t('viewer.caption.' + key), french[i],
        `viewer.caption.${key} is still the French sentence in ${lang}`
      );
    });
  }
});

// ── Measured time ───────────────────────────────────────────────────────────

test('an unmeasured stage shows a dash, never a plausible number', () => {
  // `src/api.py` sends `ms: null` for stage 13 because nothing timed it. §8: a number on
  // screen that nobody measured is worse than a blank, because the reader cannot tell
  // which one they are looking at.
  assert.strictEqual(viewer.formatMs(PROBABILITIES.ms), '—');
});

test('a measured time keeps the precision that makes it readable', () => {
  assert.strictEqual(viewer.formatMs(0.353), '0,35 ms');
  assert.strictEqual(viewer.formatMs(18.9), '18,9 ms');
});

test('a time too small to render is said to be small, not said to be zero', () => {
  // Step 8 measures ≈ 0.004 ms. Two decimals would print `0,00 ms`, which reads as free.
  assert.strictEqual(viewer.formatMs(0.004), '< 0,01 ms');
  assert.strictEqual(viewer.formatMs(0), '0 ms');
});

// ── Shape ───────────────────────────────────────────────────────────────────

test('a shape reads as the dimensions it has', () => {
  assert.strictEqual(viewer.formatShape([512, 512]), '512 × 512');
  assert.strictEqual(viewer.formatShape([128, 128, 1]), '128 × 128 × 1');
});

test('a downscaled stage reports both sizes so the cap is visible', () => {
  // §8: stage 1 is capped at 768 px for display and the UI must say so. Silently showing
  // a 768 px picture labelled 4032 px would misreport the capture.
  assert.deepStrictEqual(viewer.shapeNote(CAPTURE), {
    displayed: '576 × 768 × 4',
    original: '3024 × 4032 × 4',
  });
});

test('a stage shown at its real size has nothing to explain', () => {
  assert.strictEqual(viewer.shapeNote(CANNY), null);
});

test('a stage that was reshaped but not shrunk claims no downscale', () => {
  // Step 8 is `(128, 128, 1)` float32 and is encoded as a `(128, 128)` uint8 picture, so
  // its `display_shape` differs from its `shape` without a single pixel being lost.
  // Reporting that as "shown at 128 × 128 · original 128 × 128 × 1" announces a cap that
  // never happened — §8 wants the note where the capture really was capped, and nowhere
  // else.
  const tensor = { key: 'tensor', shape: [128, 128, 1], display_shape: [128, 128] };
  assert.strictEqual(viewer.shapeNote(tensor), null);
});

test('dimensions read as shape and dtype together', () => {
  assert.strictEqual(viewer.formatDimensions(CANNY), '512 × 512 · uint8');
});

test('a stage that is not an array says so with a dash, not with a stray separator', () => {
  // Stage 13 is a bar chart: it carries neither `shape` nor `dtype`, and joining two
  // absent fields printed a lone `·` in the panel.
  assert.strictEqual(viewer.formatDimensions(PROBABILITIES), '—');
});

// ── Parameters ──────────────────────────────────────────────────────────────

test('a parameter is printed as the config holds it, not as it reads nicely', () => {
  // §8: "Show the exact OpenCV call and the parameters read from config/preprocess.json,
  // not retyped into the UI. If someone changes a threshold, the viewer must say the new
  // number." Rounding `scale` to 0,0039 would break that promise for that one field.
  assert.deepStrictEqual(viewer.formatParams({ dtype: 'float32', scale: 0.00392156862745098 }), [
    { name: 'dtype', value: 'float32' },
    { name: 'scale', value: '0.00392156862745098' },
  ]);
});

test('a pair of numbers is printed as the grid it describes', () => {
  assert.deepStrictEqual(viewer.formatParams({ ksize: [5, 5], l2_gradient: false }), [
    { name: 'ksize', value: '5 × 5' },
    { name: 'l2_gradient', value: 'false' },
  ]);
});

test('a stage with no parameters lists none', () => {
  assert.deepStrictEqual(viewer.formatParams({}), []);
  assert.deepStrictEqual(viewer.formatParams(undefined), []);
});

test('the parameters also read as one line, for the panel that has no room for a table', () => {
  // §8 requires the config parameters on screen whatever the layout is. The overlay shows
  // them on a single line rather than dropping them to stay tidy.
  assert.strictEqual(
    viewer.formatParamsLine(CANNY.params),
    'threshold1=50 · threshold2=150 · aperture_size=3 · l2_gradient=false'
  );
});

test('a stage with no parameters contributes no line', () => {
  // Stage 1 carries `params: {}`; an empty line would leave a stray separator on screen.
  assert.strictEqual(viewer.formatParamsLine({}), '');
  assert.strictEqual(viewer.formatParamsLine(undefined), '');
});

// ── Rendering rules ─────────────────────────────────────────────────────────

test('the two stages at 128 px render without smoothing', () => {
  // §8: at 128×128 the pixel grid IS the point; smoothing it hides that the network sees
  // something coarse.
  assert.strictEqual(viewer.isPixelated({ key: 'resized' }), true);
  assert.strictEqual(viewer.isPixelated({ key: 'tensor' }), true);
  assert.strictEqual(viewer.isPixelated({ key: 'roi' }), false);
});

// ── The edge map beside the result ──────────────────────────────────────────

test('the edge map is the stage the network receives, found by key', () => {
  // The result panel shows one image: step 7, "what the network actually receives". It
  // used to be taken as `stages[stages.length - 1]`, which held only while the analyser
  // asked for `?stages=false` and got a one-element list back. Asking for all thirteen
  // makes the last stage the probability bars, which carry no image at all.
  const stages = [CAPTURE, CANNY, { key: 'resized', png_base64: 'UkVTSVpFRA==' }, PROBABILITIES];
  assert.strictEqual(viewer.edgeMapSrc(stages), 'data:image/png;base64,UkVTSVpFRA==');
});

test('a response without the resized stage shows no edge map rather than a broken one', () => {
  assert.strictEqual(viewer.edgeMapSrc([PROBABILITIES]), null);
  assert.strictEqual(viewer.edgeMapSrc([]), null);
  assert.strictEqual(viewer.edgeMapSrc(undefined), null);
});

test('the edge map reads the field the API actually sends', () => {
  // `src/api.py` names it `png_base64`. Reading `stage.png` yielded `undefined` and the
  // figure silently never appeared — a blank that looked like a design choice.
  assert.strictEqual(viewer.edgeMapSrc([{ key: 'resized', png: 'V1JPTkc=' }]), null);
});

// ── Pace ────────────────────────────────────────────────────────────────────

test('the animation lingers on Canny, the step that carries the point', () => {
  // §8: "Step 6 is the aha and should be the one the animation lingers on."
  assert.ok(viewer.dwellMs(CANNY) > viewer.dwellMs(CAPTURE));
});

test('the animation lingers on Grad-CAM, the image that answers why', () => {
  assert.ok(viewer.dwellMs({ key: 'gradcam' }) > viewer.dwellMs(CAPTURE));
});

test('an ordinary step is held about as long as it takes to look at it', () => {
  // §8 fixes the pace at ~1.1 s per step.
  assert.strictEqual(viewer.dwellMs(CAPTURE), 1100);
});

// ── Playback ────────────────────────────────────────────────────────────────

test('playback starts on the first stage', () => {
  const p = viewer.playback(13);
  assert.strictEqual(p.index, 0);
  assert.strictEqual(p.playing, false);
});

test('play then next walks forward one stage at a time', () => {
  const p = viewer.playback(13);
  p.play();
  assert.strictEqual(p.playing, true);
  p.next();
  assert.strictEqual(p.index, 1);
});

test('the film stops at the last stage instead of looping', () => {
  // A sixteen-second didactic loop running forever on a shared field laptop is noise.
  const p = viewer.playback(3);
  p.play();
  p.next();
  p.next();
  assert.strictEqual(p.index, 2);
  p.next();
  assert.strictEqual(p.index, 2, 'must not wrap round to the beginning');
  assert.strictEqual(p.playing, false, 'must stop when the film ends');
});

test('stepping back stops at the first stage', () => {
  const p = viewer.playback(13);
  p.prev();
  assert.strictEqual(p.index, 0);
});

test('taking the rail hands control to the user, so playback pauses', () => {
  const p = viewer.playback(13);
  p.play();
  p.go(5);
  assert.strictEqual(p.index, 5);
  assert.strictEqual(p.playing, false);
});

test('stepping by hand also pauses', () => {
  const p = viewer.playback(13);
  p.play();
  p.next({ manual: true });
  assert.strictEqual(p.playing, false);
});

test('a stage outside the film cannot be selected', () => {
  const p = viewer.playback(13);
  p.go(99);
  assert.strictEqual(p.index, 12);
  p.go(-4);
  assert.strictEqual(p.index, 0);
});

test('restarting rewinds and plays', () => {
  const p = viewer.playback(13);
  p.go(7);
  p.restart();
  assert.strictEqual(p.index, 0);
  assert.strictEqual(p.playing, true);
});

test('an empty response drives no playback at all', () => {
  // A failed analysis yields no stages. The viewer is hidden, but nothing here may throw.
  const p = viewer.playback(0);
  p.play();
  p.next();
  assert.strictEqual(p.index, 0);
  assert.strictEqual(p.playing, false);
});

// ── The phase window (design 1a) ────────────────────────────────────────────

test('a rank reads on two digits, because the strip aligns them', () => {
  // The strip writes the rank under each thumbnail in a monospace column. `1` beside `13`
  // shifts every label in the row by half a glyph, which is exactly what the padding of
  // the mockup exists to stop.
  assert.strictEqual(viewer.formatRank(1), '01');
  assert.strictEqual(viewer.formatRank(9), '09');
  assert.strictEqual(viewer.formatRank(13), '13');
});

test('the first phase already fills part of the bar, rather than none of it', () => {
  // The bar counts phases passed, not gaps between them. Stage 1 is on screen, so a bar
  // still empty during the first second of the film would report something untrue.
  assert.ok(viewer.progressPercent(0, 13) > 0);
  assert.strictEqual(viewer.progressPercent(0, 13), (1 / 13) * 100);
});

test('the last phase fills the bar completely', () => {
  assert.strictEqual(viewer.progressPercent(12, 13), 100);
});

test('the bar follows however many stages the response carries', () => {
  // §8 fixes thirteen stages, but `?stages=false` returns one and the count is read from
  // the response rather than typed here. A hard-coded 13 would draw an 8 % bar for a
  // single-stage answer that is, in fact, the whole of it.
  assert.strictEqual(viewer.progressPercent(0, 1), 100);
  assert.strictEqual(viewer.progressPercent(3, 8), 50);
});

test('a bar with nothing to show stays at zero instead of dividing by it', () => {
  assert.strictEqual(viewer.progressPercent(0, 0), 0);
});

test('an index outside the film cannot push the bar past full', () => {
  assert.strictEqual(viewer.progressPercent(99, 13), 100);
  assert.strictEqual(viewer.progressPercent(-4, 13), (1 / 13) * 100);
});

test('the film is finished only once it has stopped on the last stage', () => {
  // This is what swaps `Suivant` for `Voir le résultat`. Sitting on the last stage is not
  // enough on its own: `playback.next` sets `playing` false on arriving, so the two
  // conditions are read together.
  assert.strictEqual(viewer.isFinished(12, 13, false), true);
});

test('the last stage still playing is not yet finished', () => {
  assert.strictEqual(viewer.isFinished(12, 13, true), false);
});

test('pausing mid-film does not offer the way out', () => {
  // Pausing on stage 6 to read it must not replace `Suivant` with `Voir le résultat` —
  // that would strand the user seven stages short with no visible way forward.
  assert.strictEqual(viewer.isFinished(5, 13, false), false);
});

test('an empty response is never finished, because it never began', () => {
  assert.strictEqual(viewer.isFinished(0, 0, false), false);
});

test('the phase window has its own strings in all three languages', () => {
  // The strip, the progress line and the way out are written by JS, not by `data-i18n`,
  // so a missing string surfaces as the literal key inside the panel. `WI18n.t` falls
  // back to French, so each language is compared against it: these four are short but
  // none of them is spelled the same way in two languages.
  const i18n = loadI18n();
  const KEYS = ['viewer.subtitle', 'viewer.progress', 'viewer.result', 'viewer.next_label'];
  for (const key of KEYS) {
    assert.notStrictEqual(i18n.t(key), key, `no French string for ${key}`);
  }
  const french = KEYS.map((key) => i18n.t(key));
  for (const lang of ['es', 'ca']) {
    i18n.setLang(lang);
    KEYS.forEach((key, i) => {
      assert.notStrictEqual(i18n.t(key), french[i], `${key} is still the French string in ${lang}`);
    });
  }
});

test('the progress line carries the placeholders the viewer substitutes', () => {
  // `render` replaces `{n}`, `{total}` and `{t}`. A translation that drops one renders the
  // brace literally beside the bar, and the phase count silently stops being reported.
  const i18n = loadI18n();
  for (const lang of ['fr', 'es', 'ca']) {
    i18n.setLang(lang);
    const line = i18n.t('viewer.progress');
    for (const token of ['{n}', '{total}', '{t}']) {
      assert.ok(line.includes(token), `viewer.progress lost ${token} in ${lang}`);
    }
  }
});

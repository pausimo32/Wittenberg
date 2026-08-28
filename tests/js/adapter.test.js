/**
 * The mapping between the API contract (`docs/05` §3) and the shape the interface
 * renders. Pure functions, no DOM, so they are tested with `node --test tests/js/`.
 */
const test = require('node:test');
const assert = require('node:assert');
const adapter = require('../../site/js/adapter.js');

const BODY = {
  top3: [
    { class_index: 2, freq_hz: 860, p: 0.9703994 },
    { class_index: 1, freq_hz: 610, p: 0.0079200 },
    { class_index: 0, freq_hz: 420, p: 0.0064549 },
  ],
  roi_mode: 'homography',
  markers_found: 4,
  quality: { quality: 'ok', reason: null, metrics: {} },
  elapsed_ms: 111.4,
};

test('the leader carries a synthesised label, because the API sends none', () => {
  const r = adapter.mapPrediction(BODY);
  assert.strictEqual(r.success, true);
  assert.strictEqual(r.frequency.hz, 860);
  assert.strictEqual(r.frequency.label, '860 Hz');
  assert.strictEqual(r.frequency.hasLocalMetadata, false);
});

test('a probability is truncated toward zero, never rounded up', () => {
  assert.strictEqual(adapter.truncatePercent(0.6999), 69.9);
  assert.strictEqual(adapter.truncatePercent(0.9703994), 97.0);
  assert.strictEqual(adapter.tierOf(0.6999), 'mid');
  assert.strictEqual(adapter.tierOf(0.70), 'high');
  assert.strictEqual(adapter.tierOf(0.40), 'mid');
  assert.strictEqual(adapter.tierOf(0.3999), 'low');
});

test('the low tier names nobody', () => {
  const top3 = BODY.top3.map((c, i) => ({ ...c, p: [0.35, 0.34, 0.31][i] }));
  const r = adapter.mapPrediction({ ...BODY, top3 });
  assert.strictEqual(r.tier, 'low');
  assert.strictEqual(r.leader, null);
  // Still three candidates: every tier shows them (docs/05 §4).
  assert.strictEqual(r.topPredictions.length, 3);
});

test('the mid tier names two candidates as equals', () => {
  const top3 = BODY.top3.map((c, i) => ({ ...c, p: [0.51, 0.33, 0.16][i] }));
  const r = adapter.mapPrediction({ ...BODY, top3 });
  assert.strictEqual(r.tier, 'mid');
  assert.strictEqual(r.leader.hz, 860);
  assert.strictEqual(r.runnerUp.hz, 610);
});

test('top3 is re-sorted rather than trusted', () => {
  const body = { ...BODY, top3: [BODY.top3[1], BODY.top3[0], BODY.top3[2]] };
  assert.strictEqual(adapter.mapPrediction(body).frequency.hz, 860);
});

test('a centre-cropped upload reports geometry separately from confidence', () => {
  const r = adapter.mapPrediction({ ...BODY, roi_mode: 'center_crop', markers_found: 1 });
  assert.strictEqual(r.geometry.rectified, false);
  assert.strictEqual(r.geometry.markersFound, 1);
  // A wrongly rectified answer is not a low-confidence answer: both states, independently.
  assert.strictEqual(r.tier, 'high');
});

test('quality warnings survive the mapping', () => {
  const q = { quality: 'reject', reason: 'blurred', metrics: { laplacian_variance: 3.2 } };
  const r = adapter.mapPrediction({ ...BODY, quality: q });
  assert.strictEqual(r.quality.reason, 'blurred');
  assert.strictEqual(r.quality.metrics.laplacian_variance, 3.2);
});

test('an empty top3 is the low tier, not a crash and not a confident blank', () => {
  const r = adapter.mapPrediction({ ...BODY, top3: [] });
  assert.strictEqual(r.tier, 'low');
  assert.strictEqual(r.leader, null);
  assert.strictEqual(r.frequency, null);
});

test('health maps the fields the banner and the badge need', () => {
  const h = adapter.mapHealth({
    model_run: '20260828-124602',
    model: 'wittenberg_v1',
    preprocess_version: '1.1.0',
    device: 'cpu',
    classes: [{ index: 0, freq_hz: 420 }],
    synthetic: true,
    do_not_report: 'no number here may be reported.',
  });
  assert.strictEqual(h.available, true);
  assert.strictEqual(h.modelVersion, '20260828-124602');
  assert.strictEqual(h.doNotReport, 'no number here may be reported.');
});

test('a reportable run carries no banner text', () => {
  const h = adapter.mapHealth({ model_run: 'r', synthetic: false, do_not_report: null });
  assert.strictEqual(h.doNotReport, null);
});

test('an unreachable backend is not an available model', () => {
  const h = adapter.mapHealth(null);
  assert.strictEqual(h.available, false);
  assert.strictEqual(h.reachable, false);
});

test('the adapter is indifferent to how many classes exist and what they are', () => {
  // The acceptance criterion: the classes are not chosen yet.
  const body = {
    top3: [
      { class_index: 11, freq_hz: 3175, p: 0.51 },
      { class_index: 4, freq_hz: 1042, p: 0.33 },
      { class_index: 0, freq_hz: 287, p: 0.16 },
    ],
    roi_mode: 'homography',
    markers_found: 4,
    quality: { quality: 'ok', reason: null, metrics: {} },
  };
  const r = adapter.mapPrediction(body);
  assert.strictEqual(r.tier, 'mid');
  assert.strictEqual(r.leader.label, '3175 Hz');
  assert.strictEqual(r.runnerUp.label, '1042 Hz');
  assert.strictEqual(r.frequency.hasLocalMetadata, false);
});

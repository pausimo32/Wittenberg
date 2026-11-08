/**
 * When the station calls a Chladni pattern formed — ADR 0008 decision 6, check 2.
 */
const test = require('node:test');
const assert = require('node:assert');
const m = require('../../site/js/stationMotion.js');

const series = (values, step = 200) => values.map((v, i) => ({ t: i * step, v }));

test('motion is the mean absolute grey-level difference', () => {
  assert.strictEqual(m.motion([0, 10, 20], [0, 10, 26]), 2);
  assert.strictEqual(m.motion([1, 2], [1]), null);
});

test('a second under the threshold is calm', () => {
  assert.strictEqual(m.verdict(series([9, 8, 1, 1, 1, 1, 1, 1]), 2), 'calm');
});

test('two quiet samples are not yet a second', () => {
  assert.strictEqual(m.verdict(series([1, 1]), 2), null);
});

test('a formed pattern with jumping grains never goes calm, but goes steady', () => {
  // the residual jitter of the antinodes: always above the threshold, but flat
  const jitter = [5.1, 4.9, 5.2, 5.0, 4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 5.1, 4.9, 5.0, 5.1, 5.0, 4.9];
  assert.strictEqual(m.verdict(series(jitter), 2), 'steady');
});

test('a pattern that is still forming is neither calm nor steady', () => {
  const forming = [20, 18.5, 17, 15.5, 14, 12.5, 11, 9.5, 8, 6.5, 5, 4.6, 4.2, 3.8, 3.4, 3.0];
  assert.strictEqual(m.verdict(series(forming), 2), null);
});

test('the plate box is the calibration quad, shrunk, and clamped to the frame', () => {
  const box = m.plateBox([[100, 100], [500, 100], [500, 500], [100, 500]], 1920, 1080, 0.1);
  assert.deepStrictEqual(box, { x: 140, y: 140, w: 320, h: 320 });
  assert.deepStrictEqual(m.plateBox(null, 640, 480, 0.1), { x: 0, y: 0, w: 640, h: 480 });
});

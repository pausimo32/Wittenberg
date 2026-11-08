/**
 * stationMotion.js
 * Wittenberg — when the capture station calls a pattern "formed" (ADR 0008 decision 6, check 2).
 *
 * A vibrating plate is never still. Grains at the antinodes keep jumping for as long as the
 * tone plays; what stops changing once the pattern has formed is the *structure* — where the
 * nodal lines are. So the station does not ask "did any pixel move?", which a Chladni plate
 * never answers no to. It measures motion on the plate only (the calibration's quad, not the
 * room behind it), heavily downsampled so single grains average out, against the frame of one
 * second earlier rather than the previous one. Then:
 *
 * - **calm** — the motion stayed under the threshold for a full second, or
 * - **steady** — it stopped changing for three seconds: the residual jitter of a formed
 *   pattern is constant, while a pattern that is still forming or breaking up keeps falling or
 *   rising.
 *
 * Pure functions, no DOM: tested with `node --test tests/js/`.
 */
(function (root, factory) {
  var api = factory();
  root.WittenbergMotion = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var CALM_MS = 1000;
  var STEADY_MS = 3000;
  /** How flat "stopped changing" is: spread within 25 % of the mean, plus a floor for noise. */
  var STEADY_RELATIVE = 0.25;
  var STEADY_FLOOR = 0.3;

  /** Mean absolute grey-level difference between two equally sized frames (0–255). */
  function motion(a, b) {
    if (!a || !b || a.length !== b.length || !a.length) return null;
    var sum = 0;
    for (var i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
  }

  /**
   * `samples` is `[{t, v}]` in time order (ms, motion). Returns 'calm', 'steady' or null.
   * A window counts only once the samples actually span it — two samples are not a second.
   */
  function verdict(samples, threshold) {
    if (!samples.length) return null;
    var now = samples[samples.length - 1].t;

    var calm = samples.filter(function (s) { return now - s.t <= CALM_MS; });
    if (now - calm[0].t >= CALM_MS * 0.9 && calm.every(function (s) { return s.v < threshold; })) return 'calm';

    var win = samples.filter(function (s) { return now - s.t <= STEADY_MS; });
    if (now - win[0].t >= STEADY_MS * 0.9 && win.length >= 5) {
      var vs = win.map(function (s) { return s.v; });
      var max = Math.max.apply(null, vs), min = Math.min.apply(null, vs);
      var mean = vs.reduce(function (a, b) { return a + b; }, 0) / vs.length;
      if (max - min <= STEADY_RELATIVE * mean + STEADY_FLOOR) return 'steady';
    }
    return null;
  }

  /** The plate's bounding box from the calibration corners, shrunk so the markers stay out. */
  function plateBox(corners, width, height, shrink) {
    if (!corners || corners.length !== 4) return { x: 0, y: 0, w: width, h: height };
    var xs = corners.map(function (c) { return c[0]; }), ys = corners.map(function (c) { return c[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var dx = (x1 - x0) * (shrink || 0), dy = (y1 - y0) * (shrink || 0);
    x0 = Math.max(0, x0 + dx); y0 = Math.max(0, y0 + dy);
    x1 = Math.min(width, x1 - dx); y1 = Math.min(height, y1 - dy);
    if (x1 - x0 < 8 || y1 - y0 < 8) return { x: 0, y: 0, w: width, h: height };
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  return { motion: motion, verdict: verdict, plateBox: plateBox, CALM_MS: CALM_MS, STEADY_MS: STEADY_MS };
});

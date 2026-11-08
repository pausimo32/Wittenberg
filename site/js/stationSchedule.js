/**
 * stationSchedule.js
 * Wittenberg — the capture station's order of shots (ADR 0008 decision 5), and nothing else.
 *
 * A session is `rounds` rounds; one round visits every chosen class exactly once. Never sixty
 * frames of one class and then sixty of the next: anything that drifts during a session —
 * sand lost over the edges, the amplifier warming, the light — would become a function of the
 * label. Within a round the order is either the operator's own, or shuffled per round from a
 * seed that goes into the session log, so the order of any past session can be recomputed.
 *
 * Pure functions, no DOM: tested with `node --test tests/js/`.
 */
(function (root, factory) {
  var api = factory();
  root.WittenbergSchedule = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /** mulberry32: a small seeded PRNG. Deterministic across browsers and Node. */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * The classes of round `round` (0-based), in the order they are shot.
   * `shuffle` false keeps `classes` as the operator arranged them.
   */
  function roundOrder(classes, round, seed, shuffle) {
    var order = classes.slice();
    if (!shuffle) return order;
    // One generator per round, so round r's order does not depend on how many rounds came
    // before it — a resumed session recomputes the same round it stopped in.
    var next = rng((seed >>> 0) ^ Math.imul(round + 1, 0x9E3779B1));
    for (var i = order.length - 1; i > 0; i--) {
      var j = Math.floor(next() * (i + 1));
      var tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    return order;
  }

  /** Every slot of a session, `{round, index, hz}`, in shooting order. */
  function slots(classes, rounds, seed, shuffle) {
    var out = [];
    for (var r = 0; r < rounds; r++) {
      roundOrder(classes, r, seed, shuffle).forEach(function (hz, i) {
        out.push({ round: r, index: i, hz: hz });
      });
    }
    return out;
  }

  /**
   * Where a resumed session picks up: the round the least-served class has reached, counted
   * in *accepted* frames, read off the server's draft rather than remembered by a tab that may
   * have crashed. Accepted, not stored: a slot's retries store their rejections too, so stored
   * frames overcount slots. A slot that ended with no accepted frame is shot again on resume,
   * which is what a short class needs anyway.
   */
  function resumeRound(classes, acceptedPerClass) {
    var min = Infinity;
    classes.forEach(function (hz) { min = Math.min(min, acceptedPerClass[String(hz)] || 0); });
    return classes.length ? min : 0;
  }

  /** In a resumed round, a class that already has a frame for it is not shot twice. */
  function pendingInRound(order, round, acceptedPerClass) {
    return order.filter(function (hz) { return (acceptedPerClass[String(hz)] || 0) <= round; });
  }

  /** A fresh 32-bit seed for a new session. */
  function newSeed() {
    return Math.floor(Math.random() * 4294967296) >>> 0;
  }

  return {
    rng: rng, roundOrder: roundOrder, slots: slots,
    resumeRound: resumeRound, pendingInRound: pendingInRound, newSeed: newSeed
  };
});

/**
 * The capture station's order of shots — ADR 0008 decision 5, docs/05 §10 "Testing".
 */
const test = require('node:test');
const assert = require('node:assert');
const schedule = require('../../site/js/stationSchedule.js');

const CLASSES = [165, 497, 863, 1218, 2241];

test('every class appears exactly once in every round', () => {
  for (let r = 0; r < 60; r++) {
    const order = schedule.roundOrder(CLASSES, r, 1234, true);
    assert.deepStrictEqual([...order].sort((a, b) => a - b), CLASSES);
  }
});

test('the shuffled order is a pure function of the seed', () => {
  assert.deepStrictEqual(
    schedule.slots(CLASSES, 20, 42, true),
    schedule.slots(CLASSES, 20, 42, true),
  );
  assert.notDeepStrictEqual(
    schedule.slots(CLASSES, 20, 42, true),
    schedule.slots(CLASSES, 20, 43, true),
  );
});

test('a round does not depend on the rounds before it, so a resume recomputes it', () => {
  const all = schedule.slots(CLASSES, 10, 7, true).filter((s) => s.round === 6).map((s) => s.hz);
  assert.deepStrictEqual(all, schedule.roundOrder(CLASSES, 6, 7, true));
});

test('the shuffle actually varies between rounds', () => {
  const orders = new Set();
  for (let r = 0; r < 20; r++) orders.add(schedule.roundOrder(CLASSES, r, 99, true).join(','));
  assert.ok(orders.size > 5);
});

test("a fixed order is the operator's, in every round", () => {
  const mine = [2241, 165, 863];
  for (let r = 0; r < 5; r++) assert.deepStrictEqual(schedule.roundOrder(mine, r, 1, false), mine);
});

test('only the chosen classes are scheduled, never one more', () => {
  const chosen = [497, 1218];
  const hz = new Set(schedule.slots(chosen, 30, 5, true).map((s) => s.hz));
  assert.deepStrictEqual([...hz].sort((a, b) => a - b), chosen);
  assert.strictEqual(schedule.slots(chosen, 30, 5, true).length, 60);
});

test('a resumed session starts at the round the least-shot class reached', () => {
  assert.strictEqual(schedule.resumeRound(CLASSES, {}), 0);
  assert.strictEqual(
    schedule.resumeRound([165, 497], { 165: 4, 497: 3, 863: 9 }),
    3,
  );
});

test('a resumed round skips the classes that already have a frame for it', () => {
  const order = [863, 165, 497];
  assert.deepStrictEqual(schedule.pendingInRound(order, 3, { 165: 4, 497: 3, 863: 3 }), [863, 497]);
});

const audio = require('../../site/js/stationAudio.js');

test("the volume is onlinemictest.com's: gain = percent / 100", () => {
  assert.strictEqual(audio.gainFor(5), 0.05);
  assert.strictEqual(audio.gainFor(37), 0.37);
  assert.strictEqual(audio.gainFor(100), 1);
});

test('the percent is clamped to 0..100', () => {
  assert.strictEqual(audio.gainFor(150), 1);
  assert.strictEqual(audio.gainFor(-5), 0);
});

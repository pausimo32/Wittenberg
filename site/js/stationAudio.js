/**
 * stationAudio.js
 * Wittenberg — which output channel the capture station's tone leaves by.
 *
 * The plate's driver hangs off one channel of the Mac's audio output, through the amplifier.
 * A mono node connected straight to `ctx.destination` is up-mixed to *both* channels, so the
 * tone would also come out of whatever is on the other one. The station therefore routes the
 * tone through a `ChannelMergerNode` into one input only: silence on the other channel is the
 * graph's doing, not a hope about the speaker wiring. The channel is a session setting
 * (`session.json` → `station.channel`), right by default, so the log says how the plate was
 * driven.
 *
 * Verified by rendering through an `OfflineAudioContext` and measuring each channel.
 */
(function (root, factory) {
  var api = factory();
  root.WittenbergStationAudio = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var CHANNELS = ['right', 'left', 'both'];

  function normalise(channel) { return CHANNELS.indexOf(channel) >= 0 ? channel : 'right'; }

  /**
   * Connect `node` (mono) to `ctx.destination` on `channel`. Returns a function that undoes
   * the connection, so the route can be changed without rebuilding the oscillator.
   */
  function connectTone(ctx, node, channel) {
    channel = normalise(channel);
    if (channel === 'both') {
      node.connect(ctx.destination);
      return function () { node.disconnect(); };
    }
    var merger = ctx.createChannelMerger(2);
    node.connect(merger, 0, channel === 'left' ? 0 : 1);
    merger.connect(ctx.destination);
    return function () { node.disconnect(); merger.disconnect(); };
  }

  /**
   * The gain for a volume in percent, on the scale of onlinemictest.com's tone generator, which
   * the team set the rig up with: `gain = percent / 100`, linear, 0–100 in whole steps. The
   * percent is the level and nothing else — the channel is a separate setting.
   */
  function gainFor(percent) {
    var p = Math.min(100, Math.max(0, Number(percent) || 0));
    return p / 100;
  }

  return { CHANNELS: CHANNELS, normalise: normalise, connectTone: connectTone, gainFor: gainFor };
});

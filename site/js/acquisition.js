/**
 * acquisition.js
 * Wittenberg — the capture station, `docs/05` §10 and `docs/adr/0008`.
 *
 * A client of the §9 draft endpoints and nothing more: no storage path, no gate of its own.
 * What it adds is the hands — it plays each class's frequency, waits, watches the stream until
 * the pattern stops moving, grabs a frame as PNG and uploads it, then moves on.
 *
 * One slot:
 *   tone on (ramped) → dwell_s → pattern settled? → grab PNG → POST /frames (+ marker check)
 *   → accepted: next · rejected: retry up to N · never settled: skipped and logged
 *
 * Rules carried from ADR 0008, the ones easiest to "improve" away:
 * - **Round-robin.** Every round visits every chosen class once (`stationSchedule.js`).
 * - **The classifier is never a check.** Nothing here asks the model about a frame.
 * - **The tone stops whenever the run is not capturing**: pause, re-seed, error, hidden tab.
 * - **A moved rig stops the run**, it does not retry: the calibration no longer describes the
 *   frame, and retrying would only collect more frames warped to the wrong place.
 * - **The classes come from `js/classes.generated.js`**, never typed: a class the station can
 *   shoot is a class `config/classes.json` knows.
 * - **The configuration lives in the session log** (`session.json` → `station`), so a reload or
 *   a crash resumes from the server's draft with the same classes, order and seed.
 */
(function () {
  'use strict';

  var CLASSES = (window.WITTENBERG_CLASSES || []).map(function (c) { return c.hz; });
  var Sched = window.WittenbergSchedule;
  var Motion = window.WittenbergMotion;
  var SETTLE_SAMPLE_MS = 200;
  var SETTLE_WIDTH = 48;          // px across the plate: single grains average out at this scale
  var COMPARE_MS = 1000;          // motion is measured against the frame of one second earlier
  var RAMP_S = 0.08;

  function $(id) { return document.getElementById(id); }
  function t(key, vars) {
    var s = window.WI18n ? window.WI18n.t(key) : key;
    if (vars) Object.keys(vars).forEach(function (k) { s = s.split('{' + k + '}').join(String(vars[k])); });
    return s;
  }
  function esc(v) {
    return String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function reasonText(reason) {
    var key = 'sess.reason.' + reason, s = t(key);
    return s === key ? reason : s;
  }

  // ── API ───────────────────────────────────────────────────────────────────
  async function call(url, options) {
    var r;
    try { r = await fetch(url, options); }
    catch (e) { throw { status: 0, code: 'network', message: t('sess.err.network') }; }
    if (r.ok) return r.status === 204 ? null : r.json();
    var err = { status: r.status, code: '', message: r.statusText, info: {} };
    try {
      var body = await r.json();
      if (typeof body.detail === 'string') err.message = body.detail;
      else if (body.detail) { err.code = body.detail.code || ''; err.message = body.detail.message || err.message; err.info = body.detail; }
    } catch (e) { /* not JSON */ }
    throw err;
  }
  function draftUrl(id) { return '/api/drafts/' + encodeURIComponent(id); }
  function postFile(url, blob, name) {
    var form = new FormData();
    form.append('file', blob, name); // the PNG as it left the canvas, never re-encoded
    return call(url, { method: 'POST', body: form });
  }

  // ── State ─────────────────────────────────────────────────────────────────
  var draft = null;        // the server's draft
  var station = null;      // draft.session.station
  var classItems = CLASSES.map(function (hz) { return { hz: hz, on: true }; });
  var stream = null;
  var audio = null;        // { ctx, osc, gain, on }
  var wake = null;
  var runToken = 0;
  var run = { status: 'idle', round: 0, queue: null, accepted: {}, rejected: {}, skipped: {}, log: [] };

  // ── Session ───────────────────────────────────────────────────────────────
  function today() {
    var d = new Date(), m = String(d.getMonth() + 1).padStart(2, '0'), day = String(d.getDate()).padStart(2, '0');
    return { iso: d.getFullYear() + '-' + m + '-' + day, compact: d.getFullYear() + m + day };
  }

  function renderClasses() {
    var list = $('st-classes'), n = classItems.length;
    list.innerHTML = classItems.map(function (c, i) {
      return '<li class="' + (c.on ? '' : 'is-off') + '">' +
        '<span class="st-pos">' + (i + 1) + '</span>' +
        '<input type="checkbox" data-cls-toggle="' + i + '"' + (c.on ? ' checked' : '') + ' aria-label="' + c.hz + ' Hz">' +
        '<span class="st-hz">' + c.hz + ' Hz</span>' +
        '<button type="button" class="st-mini" data-cls-up="' + i + '"' + (i === 0 ? ' disabled' : '') + ' aria-label="' + esc(t('st.up')) + '">↑</button>' +
        '<button type="button" class="st-mini" data-cls-down="' + i + '"' + (i === n - 1 ? ' disabled' : '') + ' aria-label="' + esc(t('st.down')) + '">↓</button>' +
        '</li>';
    }).join('');
  }

  $('st-classes').addEventListener('click', function (e) {
    var up = e.target.getAttribute('data-cls-up'), down = e.target.getAttribute('data-cls-down');
    function swap(a, b) { var tmp = classItems[a]; classItems[a] = classItems[b]; classItems[b] = tmp; renderClasses(); }
    if (up !== null) swap(+up, +up - 1);
    if (down !== null) swap(+down, +down + 1);
  });
  $('st-classes').addEventListener('change', function (e) {
    var i = e.target.getAttribute('data-cls-toggle');
    if (i !== null) { classItems[+i].on = e.target.checked; renderClasses(); }
  });

  function msg(el, kind, html) { $(el).innerHTML = html ? '<p class="st-' + kind + '" style="margin-top:.75rem;">' + html + '</p>' : ''; }

  $('st-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    var chosen = classItems.filter(function (c) { return c.on; }).map(function (c) { return c.hz; });
    if (!chosen.length) return msg('st-session-msg', 'warn', esc(t('st.no_classes')));
    function v(id) { return $('f-' + id).value; }
    var cfg = {
      classes: chosen,
      order: document.querySelector('input[name="st-order"]:checked').value,
      seed: Number(v('seed')) >>> 0,
      rounds: Math.max(1, parseInt(v('rounds'), 10) || 1),
      reseed_every: Math.max(0, parseInt(v('reseed'), 10) || 0),
      retries: Math.max(0, parseInt(v('retries'), 10) || 0),
      settle_timeout_s: Number(v('settle_timeout')) || 15,
      settle_threshold: Number(v('settle_threshold')) || 3,
      max_marker_shift_px: Number(v('max_shift')) || 8,
      channel: Audio.normalise(v('channel')),
      unsettled: v('unsettled') === 'skip' ? 'skip' : 'capture'
    };
    $('st-create').disabled = true;
    try {
      // docs/02 §4's keys with §4's types, plus ADR 0008's `camera` and the station's own
      // configuration, so the session's order can be recomputed from its log.
      var created = await call('/api/drafts', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          session_id: v('session_id').trim(), date: v('date'), operator: v('operator').trim(), plate: v('plate').trim(),
          material: v('material'), grain_mass_g: Number(v('grain_mass_g')), distance_mm: Number(v('distance_mm')),
          lighting: v('lighting'), dwell_s: Number(v('dwell_s')), camera: 'iphone13-continuity-png',
          notes: v('notes'), station: cfg
        })
      });
      await openDraft(created.draft_id);
      msg('st-session-msg', 'ok', esc(t('st.created', { id: created.draft_id })));
    } catch (err) {
      msg('st-session-msg', 'warn', esc(err.message) + (err.code ? ' <span class="st-mono">(' + esc(err.code) + ')</span>' : ''));
    }
    $('st-create').disabled = false;
  });

  async function loadResumable() {
    var sel = $('st-resume');
    try {
      var list = (await call('/api/drafts')).drafts;
      var details = await Promise.all(list.map(function (d) { return call(draftUrl(d.draft_id)).catch(function () { return null; }); }));
      var ok = details.filter(function (d) { return d && d.session && d.session.station; });
      sel.innerHTML = ok.length
        ? ok.map(function (d) { return '<option value="' + esc(d.draft_id) + '">' + esc(d.draft_id) + ' · ' + d.accepted + ' ✓</option>'; }).join('')
        : '<option value="">' + esc(t('st.none')) + '</option>';
    } catch (e) { sel.innerHTML = '<option value="">' + esc(t('st.none')) + '</option>'; }
  }

  $('st-resume-btn').addEventListener('click', async function () {
    var id = $('st-resume').value;
    if (!id) return;
    try {
      await openDraft(id);
      msg('st-session-msg', 'ok', esc(t('st.resumed', {
        id: id, n: station.classes.length, r: station.rounds,
        o: t(station.order === 'fixed' ? 'st.order_fixed_short' : 'st.order_shuffle_short')
      })));
    } catch (err) { msg('st-session-msg', 'warn', esc(err.message)); }
  });

  async function openDraft(id) {
    draft = await call(draftUrl(id));
    station = draft.session.station;
    $('st-form').querySelectorAll('input, select, button').forEach(function (el) { el.disabled = true; });
    run = { status: 'idle', round: 0, queue: null, accepted: {}, rejected: {}, skipped: {}, log: [] };
    countFromDraft();
    run.round = Sched.resumeRound(station.classes, run.accepted);
    $('st-review-link').href = 'sessions.html#' + encodeURIComponent(id);
    fillTestHz();
    $('st-threshold-live').value = station.settle_threshold;
    showChannel();
    if (audio) ensureAudio();
    enable();
    renderRun();
    if (draft.calibration) showCalibration(draft.calibration, null);
  }

  function countFromDraft() {
    station.classes.forEach(function (hz) {
      var frames = draft.frames[String(hz)] || [];
      run.accepted[hz] = frames.filter(function (r) { return r.quality === 'ok'; }).length;
      run.rejected[hz] = frames.filter(function (r) { return r.quality !== 'ok' && r.quality !== 'pending'; }).length;
      run.skipped[hz] = run.skipped[hz] || 0;
    });
  }

  function enable() {
    var camOpen = Boolean(stream);
    $('st-devices').setAttribute('aria-disabled', draft ? 'false' : 'true');
    $('st-calib').setAttribute('aria-disabled', draft && camOpen ? 'false' : 'true');
    $('st-run').setAttribute('aria-disabled', draft && camOpen && draft.calibrated ? 'false' : 'true');
    $('st-review').setAttribute('aria-disabled', draft ? 'false' : 'true');
  }

  // ── Camera ────────────────────────────────────────────────────────────────
  // ── Camera ────────────────────────────────────────────────────────────────
  // Continuity Camera lists the iPhone twice: the camera itself and "Desk View", a cropped,
  // de-skewed view of the desk. Desk View would calibrate and then warp nothing like the plate,
  // so it is never the default.
  function isIphone(label) { return /iphone/i.test(label) && !/desk/i.test(label); }

  /** The browser's error, said as what to do about it. The raw name stays visible beside it. */
  function cameraError(e) {
    var key = {
      NotAllowedError: 'st.cam_err_denied', SecurityError: 'st.cam_err_denied',
      NotReadableError: 'st.cam_err_busy', AbortError: 'st.cam_err_busy', TrackStartError: 'st.cam_err_busy',
      NotFoundError: 'st.cam_err_missing', OverconstrainedError: 'st.cam_err_missing'
    }[e && e.name] || 'st.cam_err_other';
    return t(key) + ' (' + (e && e.name) + (e && e.message ? ': ' + e.message : '') + ')';
  }

  function camMsg(kind, text) {
    $('st-cam-msg').innerHTML = text ? '<p class="st-' + kind + '" style="margin-top:.75rem;">' + esc(text) + '</p>' : '';
  }

  async function listCameras(askPermission) {
    try {
      var devices = await navigator.mediaDevices.enumerateDevices();
      var cams = devices.filter(function (d) { return d.kind === 'videoinput'; });
      // Labels are empty until the page has been granted a camera once. Asking opens the default
      // camera for an instant — the built-in one, never the iPhone, so it cannot be "busy".
      if (askPermission && cams.length && !cams.some(function (d) { return d.label; })) {
        var probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach(function (tr) { tr.stop(); });
        cams = (await navigator.mediaDevices.enumerateDevices()).filter(function (d) { return d.kind === 'videoinput'; });
      }
      var current = $('st-cam').value, pick = null;
      cams.forEach(function (d) { if (!pick && isIphone(d.label)) pick = d.deviceId; });
      if (cams.some(function (d) { return d.deviceId === current; })) pick = current;
      $('st-cam').innerHTML = cams.length ? cams.map(function (d, i) {
        return '<option value="' + esc(d.deviceId) + '"' + (d.deviceId === pick ? ' selected' : '') + '>' +
          esc(d.label || (t('st.cam_unnamed') + ' ' + (i + 1))) + '</option>';
      }).join('') : '<option value="">' + esc(t('st.cam_none')) + '</option>';
      if (askPermission) {
        var found = cams.some(function (d) { return isIphone(d.label); });
        // Measured on 2026-11-08 (macOS 27, Chrome 156): Chrome does not declare
        // NSCameraUseContinuityCameraDeviceType, so macOS never shows it the iPhone, while
        // Safari and Photo Booth see it. On a Mac, outside Safari, that is the likely cause.
        var mac = /Macintosh/.test(navigator.userAgent);
        var safari = /Safari/.test(navigator.userAgent) && !/Chrome|Chromium|Edg|Firefox|OPR/.test(navigator.userAgent);
        camMsg(found ? '' : 'warn', found ? '' : (mac && !safari ? t('st.cam_use_safari') : t('st.cam_no_iphone')));
      }
    } catch (e) { camMsg('warn', cameraError(e)); }
  }

  $('st-cam-list').addEventListener('click', function () { listCameras(true); });
  // The iPhone appears and disappears as it is picked up, locked or moved away: keep the list live.
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', function () { listCameras(false); });
  }

  $('st-cam-open').addEventListener('click', async function () {
    var id = $('st-cam').value;
    if (stream) { stream.getTracks().forEach(function (tr) { tr.stop(); }); stream = null; }
    camMsg('', '');
    // Most specific first. A size the camera cannot give is not a reason to give up on it:
    // the later attempts drop the size, then the exactness of the device.
    var base = id ? { deviceId: { exact: id } } : {};
    var attempts = [
      Object.assign({}, base, { width: { ideal: 1920 }, height: { ideal: 1080 } }),
      base,
      id ? { deviceId: id } : true
    ];
    var lastError = null;
    for (var i = 0; i < attempts.length && !stream; i++) {
      try { stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: attempts[i] }); }
      catch (e) {
        lastError = e;
        if (e.name === 'NotAllowedError' || e.name === 'SecurityError') break; // asking again will not help
      }
    }
    if (!stream) { $('st-cam-size').textContent = ''; camMsg('warn', cameraError(lastError)); enable(); return; }
    var track = stream.getVideoTracks()[0];
    track.addEventListener('ended', function () {
      camMsg('warn', t('st.cam_lost'));
      if (run.status === 'running') pause('error', t('st.cam_lost'));
    });
    ['st-video', 'st-cam-preview'].forEach(function (vid) { $(vid).srcObject = stream; });
    var video = $('st-video');
    if (!video.videoWidth) await new Promise(function (resolve) { video.onloadedmetadata = resolve; });
    $('st-cam-size').textContent = (track.label ? track.label + ' · ' : '') + video.videoWidth + ' × ' + video.videoHeight + ' px';
    if (!isIphone(track.label)) camMsg('info', t('st.cam_not_iphone'));
    enable();
  });

  function grabPng(hz, round) {
    var v = $('st-video');
    if (!v.videoWidth) return Promise.reject({ message: t('st.cam_need') });
    var c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    return new Promise(function (resolve, reject) {
      c.toBlob(function (blob) {
        if (!blob) return reject({ message: 'toBlob returned null' });
        var stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
        resolve({ blob: blob, name: 'st_' + hz + '_r' + String(round + 1).padStart(3, '0') + '_' + stamp + '.png' });
      }, 'image/png');
    });
  }

  var settleCanvas = document.createElement('canvas');
  /** The plate only (the calibration's quad), downsampled to SETTLE_WIDTH px, in grey levels. */
  function samplePlate() {
    var v = $('st-video');
    if (!v.videoWidth) return null;
    var corners = draft && draft.calibration && draft.calibration.ok ? draft.calibration.corners : null;
    var box = Motion.plateBox(corners, v.videoWidth, v.videoHeight, 0.06);
    settleCanvas.width = SETTLE_WIDTH;
    settleCanvas.height = Math.max(8, Math.round(SETTLE_WIDTH * box.h / box.w));
    var g = settleCanvas.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.drawImage(v, box.x, box.y, box.w, box.h, 0, 0, settleCanvas.width, settleCanvas.height);
    var d = g.getImageData(0, 0, settleCanvas.width, settleCanvas.height).data, out = new Float32Array(d.length / 4);
    for (var i = 0, j = 0; i < d.length; i += 4, j++) out[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return out;
  }

  function showMotion(value) {
    $('st-motion').textContent = value === null ? '' :
      t('st.motion', { v: value.toFixed(1), thr: Number(station.settle_threshold).toFixed(1) });
  }

  // ── Tone: one oscillator for the whole session, gain-ramped on every start and stop ──
  /** The slider's percent, 0–100 in whole steps — onlinemictest.com's scale (`stationAudio.gainFor`). */
  function volumePercent() { return Math.min(100, Math.max(0, Math.round(Number($('st-volume').value) || 0))); }
  function volume() { return Audio.gainFor(volumePercent()); }

  var Audio = window.WittenbergStationAudio;

  function toneChannel() { return Audio.normalise(station && station.channel); }

  function ensureAudio() {
    var channel = toneChannel();
    if (audio) {
      // The session decides the channel; a test tone played before it was opened may differ.
      if (audio.channel !== channel) { audio.unroute(); audio.unroute = Audio.connectTone(audio.ctx, audio.gain, channel); audio.channel = channel; }
      return audio;
    }
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.type = 'sine';
    gain.gain.value = 0;
    osc.connect(gain);
    var unroute = Audio.connectTone(ctx, gain, channel);
    osc.start();
    audio = { ctx: ctx, osc: osc, gain: gain, on: false, hz: null, channel: channel, unroute: unroute };
    return audio;
  }

  function showChannel() {
    $('st-channel-now').textContent = station ? t('st.channel_now', { c: t('st.channel_' + toneChannel()) }) : '';
  }

  function toneOn(hz) {
    var a = ensureAudio(), now = a.ctx.currentTime;
    if (a.ctx.state === 'suspended') a.ctx.resume();
    a.gain.gain.cancelScheduledValues(now);
    a.gain.gain.setValueAtTime(a.gain.gain.value, now);
    a.gain.gain.linearRampToValueAtTime(0, now + RAMP_S);
    a.osc.frequency.setValueAtTime(hz, now + RAMP_S);
    a.gain.gain.linearRampToValueAtTime(volume(), now + 2 * RAMP_S);
    a.on = true; a.hz = hz;
  }

  function toneOff() {
    if (!audio || !audio.on) return;
    var now = audio.ctx.currentTime;
    audio.gain.gain.cancelScheduledValues(now);
    audio.gain.gain.setValueAtTime(audio.gain.gain.value, now);
    audio.gain.gain.linearRampToValueAtTime(0, now + RAMP_S);
    audio.on = false;
  }

  function applyVolume() {
    if (!audio || !audio.on) return;
    var now = audio.ctx.currentTime;
    audio.gain.gain.cancelScheduledValues(now);
    audio.gain.gain.setValueAtTime(audio.gain.gain.value, now);
    audio.gain.gain.linearRampToValueAtTime(volume(), now + RAMP_S);
  }

  function setVolume(percent, from) {
    var p = Math.min(100, Math.max(0, Math.round(Number(percent) || 0)));
    if (from !== 'slider') $('st-volume').value = p;
    if (from !== 'number') $('st-volume-pct').value = p;
    applyVolume();
  }

  $('st-volume').addEventListener('input', function () { setVolume(this.value, 'slider'); });
  $('st-volume-pct').addEventListener('input', function () { if (this.value !== '') setVolume(this.value, 'number'); });
  // The level drives the plate, so every settled change goes into the run's log.
  ['st-volume', 'st-volume-pct'].forEach(function (id) {
    $(id).addEventListener('change', function () { if (station) log('', t('st.log_volume', { p: volumePercent() })); });
  });

  function fillTestHz() {
    var list = station ? station.classes : CLASSES;
    $('st-test-hz').innerHTML = list.map(function (hz) { return '<option value="' + hz + '">' + hz + ' Hz</option>'; }).join('');
  }
  $('st-test-tone').addEventListener('click', function () { toneOn(Number($('st-test-hz').value)); });
  $('st-test-stop').addEventListener('click', toneOff);

  // ── Calibration ───────────────────────────────────────────────────────────
  $('st-calib-btn').addEventListener('click', async function () {
    var btn = this;
    btn.disabled = true;
    $('st-calib-out').innerHTML = '<p class="st-muted" style="margin-top:.75rem;">' + esc(t('sess.cal.working')) + '</p>';
    try {
      toneOff(); // a vibrating plate blurs the markers
      var shot = await grabPng(0, -1);
      var answer = await postFile(draftUrl(draft.draft_id) + '/calibration', shot.blob, 'calibration.png');
      draft = await call(draftUrl(draft.draft_id));
      showCalibration(answer.calibration, answer.preview);
      enable();
    } catch (err) {
      $('st-calib-out').innerHTML = '<p class="st-warn" style="margin-top:.75rem;">' + esc(err.message) + '</p>';
    }
    btn.disabled = false;
  });

  function showCalibration(c, preview) {
    $('st-calib-out').innerHTML =
      '<p style="margin-top:.75rem;">' + esc(t('sess.cal.found', { n: c.n_markers })) + '</p>' +
      (c.ok ? '<p class="st-ok" style="margin-top:.5rem;">' + esc(t('sess.cal.ok')) + (c.px_per_mm ? ' · <span class="st-mono">' + c.px_per_mm.toFixed(2) + ' px/mm</span>' : '') + '</p>'
            : '<p class="st-warn" style="margin-top:.5rem;">' + esc(t('sess.cal.failed')) + '</p>') +
      (c.flags && c.flags.length ? '<div class="st-warn" style="margin-top:.5rem;"><strong>' + esc(t('sess.cal.flags')) + '</strong><ul style="padding-left:1.2rem;list-style:disc;">' +
        c.flags.map(function (f) { return '<li>' + esc(f) + '</li>'; }).join('') + '</ul></div>' : '') +
      (preview ? '<img class="st-preview" alt="' + esc(t('sess.cal.preview_alt')) + '" src="data:image/png;base64,' + preview + '">' : '');
  }

  // ── The run ───────────────────────────────────────────────────────────────
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function alive(token) { return token === runToken && run.status === 'running'; }

  function phase(text) { $('st-phase').textContent = text || ''; }
  function alertMsg(kind, text) { $('st-alert').innerHTML = text ? '<p class="st-' + kind + '">' + esc(text) + '</p>' : ''; }

  function log(kind, text) {
    var time = new Date().toLocaleTimeString();
    run.log.unshift({ kind: kind, text: time + '  ' + text });
    if (run.log.length > 300) run.log.pop();
    $('st-log').innerHTML = run.log.map(function (l) { return '<li class="' + l.kind + '">' + esc(l.text) + '</li>'; }).join('');
  }

  function renderRun() {
    if (!station) return;
    var total = station.classes.length * station.rounds, done = 0;
    station.classes.forEach(function (hz) { done += Math.min(run.accepted[hz] || 0, station.rounds); });
    $('st-progress').style.width = (100 * done / total).toFixed(1) + '%';
    $('st-round').textContent = t('st.round', { r: Math.min(run.round + 1, station.rounds), n: station.rounds });
    $('st-table').innerHTML = '<thead><tr><th>' + esc(t('st.col_class')) + '</th><th>' + esc(t('st.col_ok')) + '</th><th>' +
      esc(t('st.col_rej')) + '</th><th>' + esc(t('st.col_skip')) + '</th></tr></thead><tbody>' +
      station.classes.map(function (hz) {
        return '<tr><td class="st-mono">' + hz + ' Hz</td><td class="st-mono">' + (run.accepted[hz] || 0) + ' / ' + station.rounds +
          '</td><td class="st-mono">' + (run.rejected[hz] || 0) + '</td><td class="st-mono">' + (run.skipped[hz] || 0) + '</td></tr>';
      }).join('') + '</tbody>';
    var s = run.status;
    $('st-start').classList.toggle('st-hidden', s !== 'idle');
    $('st-pause').classList.toggle('st-hidden', s !== 'running');
    $('st-resume-run').classList.toggle('st-hidden', s !== 'paused');
    $('st-stop').classList.toggle('st-hidden', s !== 'running' && s !== 'paused');
  }

  /**
   * Dwell, then watch the plate until the pattern has formed — `stationMotion.js` says what
   * "formed" means on a plate whose grains never stop jumping. Resolves 'calm' or 'steady',
   * false when the timeout ran out first, null when the run stopped meanwhile.
   */
  async function waitForPattern(hz, token, withDwell) {
    if (withDwell) {
      var end = performance.now() + station_dwell() * 1000;
      while (performance.now() < end) {
        if (!alive(token)) return null;
        phase(t('st.ph_dwell', { s: Math.ceil((end - performance.now()) / 1000) }));
        await sleep(100);
      }
    }
    phase(t('st.ph_settle'));
    var deadline = performance.now() + station.settle_timeout_s * 1000;
    var frames = [], samples = [];
    while (performance.now() < deadline) {
      await sleep(SETTLE_SAMPLE_MS);
      if (!alive(token)) return null;
      var now = performance.now(), cur = samplePlate();
      if (!cur) continue;
      frames.push({ t: now, f: cur });
      while (frames.length > 1 && now - frames[1].t >= COMPARE_MS) frames.shift();
      if (now - frames[0].t < COMPARE_MS * 0.9) continue;
      var value = Motion.motion(cur, frames[0].f);
      if (value === null) continue;
      samples.push({ t: now, v: value });
      lastMotion = value;
      showMotion(value);
      var verdict = Motion.verdict(samples, station.settle_threshold);
      if (verdict) return verdict;
    }
    return false;
  }

  var lastMotion = null;

  function station_dwell() { return Number(draft.session.dwell_s) || 0; }

  /**
   * One slot. Returns 'ok', 'rejected' (only rejections were stored), 'skipped' (nothing was
   * stored at all — never settled, or every attempt refused before storing), or null if the
   * run stopped meanwhile. A slot that ends with nothing must still be counted somewhere.
   */
  async function shoot(hz, token) {
    var stored = false;
    $('st-now').textContent = hz + ' Hz';
    toneOn(hz);
    var settled = await waitForPattern(hz, token, true);
    if (settled === null) return null;
    if (!settled) {
      // The gate, not this heuristic, decides whether a frame is usable; skipping is opt-in.
      if (station.unsettled === 'skip') { log('bad', t('st.log_skip', { hz: hz })); return 'skipped'; }
      log('bad', t('st.log_unsettled', { hz: hz, v: lastMotion === null ? '?' : lastMotion.toFixed(1) }));
    }

    for (var attempt = 0; attempt <= station.retries; attempt++) {
      if (attempt > 0) {
        var again = await waitForPattern(hz, token, false);
        if (again === null) return null;
        if (!again && station.unsettled === 'skip') { log('bad', t('st.log_skip', { hz: hz })); return stored ? 'rejected' : 'skipped'; }
      }
      phase(t('st.ph_upload'));
      var record;
      try {
        var shot = await grabPng(hz, run.round);
        record = await postFile(draftUrl(draft.draft_id) + '/frames?freq_hz=' + hz +
          '&max_marker_shift_px=' + station.max_marker_shift_px, shot.blob, shot.name);
      } catch (err) {
        if (!alive(token)) return null;
        if (err.code === 'rig_moved') {
          pause('moved', err.message);
          log('bad', t('st.log_moved', { hz: hz }));
          return null;
        }
        if (err.code === 'duplicate_content') { log('bad', t('st.log_dup', { hz: hz })); continue; }
        pause('error', err.message);
        return null;
      }
      if (record.quality === 'ok') {
        run.accepted[hz] = (run.accepted[hz] || 0) + 1;
        log('good', t('st.log_ok', { hz: hz, n: run.accepted[hz] }));
        return 'ok';
      }
      stored = true;
      run.rejected[hz] = (run.rejected[hz] || 0) + 1;
      log('bad', t('st.log_rej', { hz: hz, reason: reasonText(record.reason || record.quality) }));
      renderRun();
      if (attempt < station.retries) phase(t('st.ph_retry', { i: attempt + 1, n: station.retries }));
    }
    return stored ? 'rejected' : 'skipped';
  }

  async function loop(token) {
    while (run.round < station.rounds) {
      if (!run.queue) {
        run.queue = Sched.pendingInRound(
          Sched.roundOrder(station.classes, run.round, station.seed, station.order !== 'fixed'),
          run.round, run.accepted);
      }
      while (run.queue.length) {
        var hz = run.queue[0];
        var outcome = await shoot(hz, token);
        if (outcome === null || !alive(token)) return;   // paused or stopped: this slot is redone
        if (outcome === 'skipped') run.skipped[hz] = (run.skipped[hz] || 0) + 1;
        run.queue.shift();
        renderRun();
      }
      run.round++;
      run.queue = null;
      renderRun();
      if (station.reseed_every && run.round % station.reseed_every === 0 && run.round < station.rounds) {
        pause('reseed', t('st.alert_reseed', { r: run.round, g: draft.session.grain_mass_g }));
        log('', t('st.log_reseed', { r: run.round }));
        return;
      }
    }
    finish();
  }

  async function holdWake() {
    try {
      if ('wakeLock' in navigator) {
        wake = await navigator.wakeLock.request('screen');
        wake.addEventListener('release', function () { if (run.status === 'running') alertMsg('warn', t('st.alert_wake')); });
      } else alertMsg('warn', t('st.alert_wake'));
    } catch (e) { alertMsg('warn', t('st.alert_wake')); }
  }
  function releaseWake() { if (wake) { wake.release().catch(function () {}); wake = null; } }

  function start() {
    if (!draft || !draft.calibrated || !stream) return;
    ensureAudio();          // the user's click is the gesture the browser wants for audio
    run.status = 'running';
    runToken++;
    alertMsg('', '');
    holdWake();
    log('', t(run.round > 0 || run.log.length ? 'st.log_resume' : 'st.log_start', { r: run.round + 1 }) + ' · ' + t('st.log_volume', { p: volumePercent() }));
    renderRun();
    loop(runToken);
  }

  function pause(why, text) {
    if (run.status !== 'running') return;
    runToken++;
    toneOff();
    releaseWake();
    run.status = 'paused';
    phase(t('st.ph_paused'));
    var kind = why === 'reseed' ? 'info' : 'warn';
    alertMsg(kind, text || t('st.alert_paused'));
    if (why === 'moved') alertMsg('warn', t('st.alert_moved') + ' ' + text);
    if (why === 'hidden') log('bad', t('st.log_hidden'));
    if (why === 'user') log('', t('st.log_pause'));
    renderRun();
  }

  function finish() {
    runToken++;
    toneOff();
    releaseWake();
    run.status = 'done';
    $('st-now').textContent = '✓';
    phase(t('st.ph_done'));
    var ok = 0;
    station.classes.forEach(function (hz) { ok += run.accepted[hz] || 0; });
    alertMsg('ok', t('st.done', { ok: ok }));
    log('good', t('st.done', { ok: ok }));
    renderRun();
  }

  // Tuned live while watching the motion readout. It changes when a frame is taken, never what
  // is stored, so it is logged rather than written back into the session log.
  $('st-threshold-live').addEventListener('change', function () {
    var v = Number(this.value);
    if (!station || !(v > 0)) return;
    station.settle_threshold = v;
    log('', t('st.log_threshold', { v: v.toFixed(1) }));
    if (lastMotion !== null) showMotion(lastMotion);
  });

  $('st-start').addEventListener('click', start);
  $('st-resume-run').addEventListener('click', start);
  $('st-pause').addEventListener('click', function () { pause('user'); });
  $('st-stop').addEventListener('click', function () {
    if (run.status === 'running') pause('user');
    finish();
  });

  // A background tab throttles timers, and nobody is watching it: stop rather than drift.
  document.addEventListener('visibilitychange', function () {
    if (document.hidden && run.status === 'running') pause('hidden', t('st.alert_hidden'));
  });
  window.addEventListener('beforeunload', toneOff);

  // ── Boot ──────────────────────────────────────────────────────────────────
  async function checkServer() {
    var box = $('st-server');
    try {
      var r = await fetch('/api/health');
      if (!r.ok) throw new Error();
      var mode = (await r.json()).mode;
      box.innerHTML = mode === 'capture' ? '<p class="st-info" style="margin-bottom:1rem;">' + esc(t('sess.capture_mode')) + '</p>' : '';
    } catch (e) {
      box.innerHTML = '<p class="st-warn" style="margin-bottom:1rem;">' + t('sess.server_down') + '</p>';
    }
  }

  function retranslate() {
    showChannel();
    renderClasses();
    renderRun();
    if (draft && draft.calibration) showCalibration(draft.calibration, null);
    checkServer();
  }

  document.addEventListener('wittenberg:langchange', retranslate);
  document.addEventListener('DOMContentLoaded', function () {
    var d = today();
    $('f-session_id').value = d.compact + '-01';
    $('f-date').value = d.iso;
    $('f-seed').value = Sched.newSeed();
    renderClasses();
    fillTestHz();
    enable();
    checkServer();
    loadResumable();
    if (navigator.mediaDevices) listCameras(false);
  });
})();

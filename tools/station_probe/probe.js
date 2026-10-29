/**
 * Station feasibility probe (ADR 0008). Throwaway: every block writes what it measured into
 * `report`, and "Save report" posts it to `runs/station-probe-<ts>/probe.json`.
 *
 * The calibration and the gate are the server's (`/api/drafts`, unchanged). This file only
 * measures what a browser can see: the camera, the audio path, and how fast a pattern forms.
 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var report = {
    probe_version: 1,
    started: new Date().toISOString(),
    environment: {
      user_agent: navigator.userAgent,
      wake_lock_supported: 'wakeLock' in navigator,
      image_capture_supported: 'ImageCapture' in window
    }
  };
  var stream = null;
  var track = null;
  var draftId = null;
  var lastCalibration = null;
  var audio = null;           // { ctx, osc, gain }

  function show(id, value) {
    $(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    $('report').textContent = JSON.stringify(report, null, 2);
  }

  function fail(id, err) {
    show(id, 'ERROR: ' + (err && err.message ? err.message : String(err)));
  }

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function round(x, n) { var k = Math.pow(10, n || 2); return Math.round(x * k) / k; }

  function json(response) {
    return response.json().then(function (body) {
      if (!response.ok) {
        var e = new Error(response.status + ' ' + JSON.stringify(body));
        e.body = body;
        e.status = response.status;
        throw e;
      }
      return body;
    });
  }

  // ------------------------------------------------------------------ context
  fetch('/probe/context').then(json).then(function (ctx) {
    report.context = ctx;
    $('run-dir').textContent = ctx.run_dir;
    show('report', report);
  }).catch(function (e) { fail('report', e); });

  // ------------------------------------------------------------------ 1 cameras
  function settingsOf(t) {
    var s = t.getSettings();
    return {
      label: t.label,
      settings: s,
      video_size: [$('video').videoWidth, $('video').videoHeight]
    };
  }

  function openCamera(constraints) {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    return navigator.mediaDevices.getUserMedia({ video: constraints, audio: false }).then(function (s) {
      stream = s;
      track = s.getVideoTracks()[0];
      $('video').srcObject = s;
      return new Promise(function (resolve) {
        $('video').onloadedmetadata = function () { resolve(track); };
      });
    });
  }

  $('cam-list').onclick = function () {
    navigator.mediaDevices.getUserMedia({ video: true }).then(function (s) {
      s.getTracks().forEach(function (t) { t.stop(); });
      return navigator.mediaDevices.enumerateDevices();
    }).then(function (devices) {
      var cams = devices.filter(function (d) { return d.kind === 'videoinput'; });
      var mics = devices.filter(function (d) { return d.kind === 'audioinput'; });
      $('cam-select').innerHTML = '';
      cams.forEach(function (d) {
        var o = document.createElement('option');
        o.value = d.deviceId; o.textContent = d.label || d.deviceId;
        if (/iphone/i.test(d.label)) o.selected = true;
        $('cam-select').appendChild(o);
      });
      $('mic-select').innerHTML = '';
      mics.forEach(function (d) {
        var o = document.createElement('option');
        o.value = d.deviceId; o.textContent = d.label || d.deviceId;
        $('mic-select').appendChild(o);
      });
      report.cameras = cams.map(function (d) { return d.label; });
      report.microphones = mics.map(function (d) { return d.label; });
      show('b1-out', { cameras: report.cameras, microphones: report.microphones });
    }).catch(function (e) { fail('b1-out', e); });
  };

  $('cam-open').onclick = function () {
    var id = $('cam-select').value;
    openCamera({ deviceId: { exact: id }, width: { ideal: 4032 }, height: { ideal: 3024 } })
      .then(function (t) {
        report.camera = {
          capabilities: t.getCapabilities ? t.getCapabilities() : null,
          at_max: settingsOf(t)
        };
        show('b1-out', report.camera);
      }).catch(function (e) { fail('b1-out', e); });
  };

  $('cam-formats').onclick = function () {
    var id = $('cam-select').value;
    var tries = [
      { name: '16:9 ideal 1920x1080', c: { width: { ideal: 1920 }, height: { ideal: 1080 } } },
      { name: '16:9 ideal 3840x2160', c: { width: { ideal: 3840 }, height: { ideal: 2160 } } },
      { name: '4:3 ideal 1920x1440', c: { width: { ideal: 1920 }, height: { ideal: 1440 } } },
      { name: '4:3 aspectRatio 4/3, width 4032', c: { aspectRatio: { exact: 4 / 3 }, width: { ideal: 4032 } } }
    ];
    var results = [];
    var chain = Promise.resolve();
    tries.forEach(function (t) {
      chain = chain.then(function () {
        var c = Object.assign({ deviceId: { exact: id } }, t.c);
        return openCamera(c).then(function (tr) {
          var s = settingsOf(tr);
          results.push({ asked: t.name, got: s.video_size, frame_rate: s.settings.frameRate });
        }).catch(function (e) { results.push({ asked: t.name, error: e.name + ': ' + e.message }); });
      });
    });
    chain.then(function () {
      report.camera = report.camera || {};
      report.camera.formats = results;
      // Leave the stream at maximum resolution: the remaining blocks measure on that.
      return openCamera({ deviceId: { exact: id }, width: { ideal: 4032 }, height: { ideal: 3024 } });
    }).then(function () { show('b1-out', results); })
      .catch(function (e) { fail('b1-out', e); });
  };

  // ------------------------------------------------------------------ frame helpers
  function grabCanvas(maxWidth) {
    var v = $('video');
    var w = v.videoWidth, h = v.videoHeight;
    if (!w) throw new Error('no video: open the camera in block 1 first');
    var scale = maxWidth && w > maxWidth ? maxWidth / w : 1;
    var c = document.createElement('canvas');
    c.width = Math.round(w * scale); c.height = Math.round(h * scale);
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
    return c;
  }

  function grabPng() {
    var t0 = performance.now();
    var c = grabCanvas(0);
    return new Promise(function (resolve, reject) {
      c.toBlob(function (blob) {
        if (!blob) return reject(new Error('toBlob returned null'));
        resolve({ blob: blob, width: c.width, height: c.height, ms: round(performance.now() - t0, 1) });
      }, 'image/png');
    });
  }

  function gray(c) {
    var d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    var g = new Float32Array(c.width * c.height);
    for (var i = 0, j = 0; i < d.length; i += 4, j++) g[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return g;
  }

  // ------------------------------------------------------------------ 2 locks
  function leftQuarterLuma() {
    var c = grabCanvas(320);
    var g = gray(c), w = c.width, h = c.height, s = 0, n = 0;
    for (var y = 0; y < h; y++) for (var x = 0; x < Math.floor(w / 4); x++) { s += g[y * w + x]; n++; }
    return s / n;
  }

  function lumaTest(label) {
    var series = [];
    var t0 = performance.now();
    return new Promise(function (resolve) {
      var timer = setInterval(function () {
        var t = (performance.now() - t0) / 1000;
        $('lock-clock').textContent = t < 3 ? 'wait…' : t < 7 ? 'COVER the right half now' : 'uncover';
        series.push([round(t, 2), round(leftQuarterLuma(), 2)]);
        if (t >= 10) {
          clearInterval(timer);
          $('lock-clock').textContent = '';
          var before = series.filter(function (p) { return p[0] < 3; }).map(function (p) { return p[1]; });
          var during = series.filter(function (p) { return p[0] >= 4 && p[0] < 7; }).map(function (p) { return p[1]; });
          var mean = function (a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; };
          var b = mean(before), d = mean(during);
          resolve({
            label: label,
            settings: track.getSettings(),
            left_quarter_luma_before: round(b, 2),
            left_quarter_luma_covered: round(d, 2),
            change_percent: round(100 * (d - b) / b, 2),
            series: series
          });
        }
      }, 200);
    });
  }

  $('lock-auto').onclick = function () {
    if (!track) return fail('b2-out', new Error('open the camera in block 1 first'));
    lumaTest('auto').then(function (r) {
      report.lock = report.lock || {};
      report.lock.auto = r;
      show('b2-out', Object.assign({}, r, { series: '(' + r.series.length + ' samples, in report)' }));
    });
  };

  $('lock-apply').onclick = function () {
    if (!track) return fail('b2-out', new Error('open the camera in block 1 first'));
    var caps = track.getCapabilities ? track.getCapabilities() : {};
    var wanted = {};
    ['exposureMode', 'focusMode', 'whiteBalanceMode'].forEach(function (k) {
      var modes = caps[k] || [];
      if (modes.indexOf('manual') >= 0) wanted[k] = 'manual';
      else if (modes.indexOf('none') >= 0) wanted[k] = 'none';
    });
    var before = track.getSettings();
    var attempt = Object.keys(wanted).length
      ? track.applyConstraints({ advanced: [wanted] })
      : Promise.reject(new Error('no lockable mode in getCapabilities()'));
    attempt.then(function () {
      return { ok: true };
    }, function (e) {
      return { ok: false, error: e.name + ': ' + e.message };
    }).then(function (res) {
      var after = track.getSettings();
      report.lock = report.lock || {};
      report.lock.apply = {
        capabilities: {
          exposureMode: caps.exposureMode || null,
          focusMode: caps.focusMode || null,
          whiteBalanceMode: caps.whiteBalanceMode || null,
          exposureTime: caps.exposureTime || null,
          exposureCompensation: caps.exposureCompensation || null
        },
        asked: wanted,
        result: res,
        before: before,
        after: after
      };
      show('b2-out', report.lock.apply);
    });
  };

  $('lock-test').onclick = function () {
    if (!track) return fail('b2-out', new Error('open the camera in block 1 first'));
    lumaTest('locked').then(function (r) {
      report.lock = report.lock || {};
      report.lock.locked = r;
      show('b2-out', Object.assign({}, r, { series: '(' + r.series.length + ' samples, in report)' }));
    });
  };

  // ------------------------------------------------------------------ draft
  function ensureDraft() {
    if (draftId) return Promise.resolve(draftId);
    var today = report.context.today;
    var log = {
      session_id: today + '-99',
      date: today.slice(0, 4) + '-' + today.slice(4, 6) + '-' + today.slice(6, 8),
      operator: 'station-probe',
      plate: 'probe',
      material: 'sand',
      grain_mass_g: 10,
      distance_mm: Number($('distance-mm').value) || 0,
      lighting: 'lamp_diffuse',
      dwell_s: 0,
      notes: 'ADR 0008 feasibility probe - throwaway draft',
      camera: 'iphone13-continuity-png'
    };
    return fetch('/api/drafts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(log)
    }).then(json).then(function (d) {
      draftId = d.draft_id;
      return draftId;
    });
  }

  function postFile(path, blob, name) {
    var fd = new FormData();
    fd.append('file', blob, name);
    var t0 = performance.now();
    return fetch(path, { method: 'POST', body: fd }).then(json).then(function (body) {
      return { body: body, ms: round(performance.now() - t0, 1) };
    });
  }

  // ------------------------------------------------------------------ 3 calibration
  function calibrateOnce() {
    return ensureDraft().then(function (id) {
      return grabPng().then(function (png) {
        return postFile('/api/drafts/' + id + '/calibration', png.blob, 'calibration.png')
          .then(function (r) { return { png: png, r: r }; });
      });
    });
  }

  $('calib-one').onclick = function () {
    calibrateOnce().then(function (x) {
      var c = x.r.body.calibration;
      $('calib-preview').src = 'data:image/png;base64,' + x.r.body.preview;
      if (c.ok) lastCalibration = { info: c, size: [x.png.width, x.png.height] };
      report.calibration = {
        info: c,
        frame_size: [x.png.width, x.png.height],
        png_bytes: x.png.blob.size,
        grab_ms: x.png.ms,
        upload_and_calibrate_ms: x.r.ms,
        marker_side_mm: Number($('marker-mm').value) || null,
        distance_mm: Number($('distance-mm').value) || null
      };
      show('b3-out', report.calibration);
    }).catch(function (e) { fail('b3-out', e); });
  };

  $('calib-jitter').onclick = function () {
    var runs = [];
    var chain = Promise.resolve();
    for (var i = 0; i < 8; i++) {
      chain = chain.then(function () {
        return calibrateOnce().then(function (x) { runs.push(x.r.body.calibration); return sleep(1000); });
      });
    }
    chain.then(function () {
      var good = runs.filter(function (c) { return c.ok; });
      var ref = good[0];
      var shifts = good.map(function (c) {
        return Math.max.apply(null, c.corners.map(function (p, k) {
          return Math.hypot(p[0] - ref.corners[k][0], p[1] - ref.corners[k][1]);
        }));
      });
      var maxPx = shifts.length ? Math.max.apply(null, shifts) : null;
      report.jitter = {
        frames: runs.length,
        detected: good.length,
        n_markers: runs.map(function (c) { return c.n_markers; }),
        max_corner_shift_px: maxPx === null ? null : round(maxPx, 3),
        max_corner_shift_mm: maxPx === null || !ref ? null : round(maxPx / ref.px_per_mm, 4),
        px_per_mm: good.map(function (c) { return round(c.px_per_mm, 4); }),
        shifts_px: shifts.map(function (s) { return round(s, 3); })
      };
      if (ref) lastCalibration = { info: good[good.length - 1], size: [$('video').videoWidth, $('video').videoHeight] };
      show('b3-out', report.jitter);
    }).catch(function (e) { fail('b3-out', e); });
  };

  // ------------------------------------------------------------------ 4 one frame
  $('frame-one').onclick = function () {
    var name = 'probe-' + Date.now() + '.png';
    ensureDraft().then(function (id) {
      return grabPng().then(function (png) {
        return postFile('/api/drafts/' + id + '/frames?freq_hz=1', png.blob, name)
          .then(function (r) { return { png: png, r: r }; });
      });
    }).then(function (x) {
      report.frames = report.frames || [];
      var entry = {
        verdict: x.r.body,
        frame_size: [x.png.width, x.png.height],
        png_bytes: x.png.blob.size,
        grab_ms: x.png.ms,
        upload_and_gate_ms: x.r.ms
      };
      report.frames.push(entry);
      show('b4-out', entry);
    }).catch(function (e) { fail('b4-out', e); });
  };

  // ------------------------------------------------------------------ 5 framing
  $('framing').onclick = function () {
    if (!lastCalibration) return fail('b5-out', new Error('no successful calibration yet (block 3)'));
    var c = lastCalibration.info, w = lastCalibration.size[0], h = lastCalibration.size[1];
    var ppm = c.px_per_mm;
    var xs = c.corners.map(function (p) { return p[0]; });
    var ys = c.corners.map(function (p) { return p[1]; });
    var margins = {
      left_mm: Math.min.apply(null, xs) / ppm,
      right_mm: (w - Math.max.apply(null, xs)) / ppm,
      top_mm: Math.min.apply(null, ys) / ppm,
      bottom_mm: (h - Math.max.apply(null, ys)) / ppm
    };
    Object.keys(margins).forEach(function (k) { margins[k] = round(margins[k], 1); });
    var markerMm = Number($('marker-mm').value) || null;
    report.framing = {
      frame_px: [w, h],
      aspect: round(w / h, 4),
      px_per_mm: round(ppm, 4),
      frame_mm: [round(w / ppm, 1), round(h / ppm, 1)],
      plate_corner_to_frame_edge_mm: margins,
      // docs/02 §1.2: each marker plus its quiet zone sits outside the plate corner.
      needed_per_side_mm: markerMm ? round(1.2 * markerMm, 1) : null,
      roi_needs_px_per_mm: 2.84,
      distance_mm: Number($('distance-mm').value) || null,
      marker_side_mm: markerMm
    };
    show('b5-out', report.framing);
  };

  // ------------------------------------------------------------------ 6 tone
  $('tone-vol').oninput = function () {
    $('tone-vol-val').textContent = Number(this.value).toFixed(2);
    if (audio) audio.gain.gain.setTargetAtTime(Number(this.value), audio.ctx.currentTime, 0.02);
  };

  function toneOn(hz) {
    if (audio) toneOff();
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = hz;
    gain.gain.setValueAtTime(0, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(Number($('tone-vol').value), ctx.currentTime + 0.1);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    audio = { ctx: ctx, osc: osc, gain: gain, hz: hz };
    report.audio = report.audio || {};
    report.audio.sample_rate = ctx.sampleRate;
    report.audio.base_latency_s = ctx.baseLatency || null;
    report.audio.output_latency_s = ctx.outputLatency || null;
  }

  function toneOff() {
    if (!audio) return;
    var a = audio;
    audio = null;
    var t = a.ctx.currentTime;
    a.gain.gain.cancelScheduledValues(t);
    a.gain.gain.setValueAtTime(a.gain.gain.value, t);
    a.gain.gain.linearRampToValueAtTime(0, t + 0.1);
    a.osc.stop(t + 0.12);
    setTimeout(function () { a.ctx.close(); }, 300);
  }

  $('tone-on').onclick = function () { toneOn(Number($('tone-hz').value)); show('b6-out', 'tone on at ' + $('tone-hz').value + ' Hz'); };
  $('tone-off').onclick = function () { toneOff(); show('b6-out', 'tone off'); };

  $('mic-fft').onclick = function () {
    if (!audio) return fail('b6-out', new Error('turn the tone on first'));
    var played = audio.hz;
    var micId = $('mic-select').value;
    navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: micId ? { exact: micId } : undefined,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false
      }
    }).then(function (micStream) {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var src = ctx.createMediaStreamSource(micStream);
      var an = ctx.createAnalyser();
      an.fftSize = 32768;
      an.smoothingTimeConstant = 0;
      src.connect(an);
      var bins = new Float32Array(an.frequencyBinCount);
      var peaks = [];
      var binHz = ctx.sampleRate / an.fftSize;
      return sleep(500).then(function () {
        var reads = [];
        for (var i = 0; i < 4; i++) reads.push(i);
        return reads.reduce(function (p) {
          return p.then(function () {
            an.getFloatFrequencyData(bins);
            var lo = Math.floor(20 / binHz), best = lo;
            for (var k = lo; k < bins.length; k++) if (bins[k] > bins[best]) best = k;
            // Parabolic interpolation around the peak bin, in dB.
            var a = bins[best - 1], b = bins[best], c = bins[best + 1];
            var off = (a - 2 * b + c) !== 0 ? 0.5 * (a - c) / (a - 2 * b + c) : 0;
            peaks.push({ peak_hz: round((best + off) * binHz, 2), peak_db: round(b, 1) });
            return sleep(400);
          });
        }, Promise.resolve());
      }).then(function () {
        micStream.getTracks().forEach(function (t) { t.stop(); });
        ctx.close();
        var r = {
          played_hz: played,
          microphone: $('mic-select').selectedOptions[0] ? $('mic-select').selectedOptions[0].textContent : null,
          sample_rate: ctx.sampleRate,
          bin_hz: round(binHz, 3),
          reads: peaks,
          max_error_hz: round(Math.max.apply(null, peaks.map(function (p) { return Math.abs(p.peak_hz - played); })), 2),
          volume: Number($('tone-vol').value)
        };
        report.audio = report.audio || {};
        report.audio.fft = report.audio.fft || [];
        report.audio.fft.push(r);
        show('b6-out', r);
      });
    }).catch(function (e) { fail('b6-out', e); });
  };

  // ------------------------------------------------------------------ 6b formation curve
  function plot(series, toneAt) {
    var c = $('settle-plot'), g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    if (!series.length) return;
    var tMax = series[series.length - 1][0];
    var yMax = Math.max.apply(null, series.map(function (p) { return p[1]; })) || 1;
    g.strokeStyle = '#888';
    g.beginPath();
    g.moveTo(toneAt / tMax * c.width, 0); g.lineTo(toneAt / tMax * c.width, c.height);
    g.stroke();
    g.strokeStyle = '#2f81f7';
    g.beginPath();
    series.forEach(function (p, i) {
      var x = p[0] / tMax * c.width, y = c.height - p[1] / yMax * (c.height - 10);
      if (i) g.lineTo(x, y); else g.moveTo(x, y);
    });
    g.stroke();
  }

  $('settle-run').onclick = function () {
    if (!track) return fail('b6-out', new Error('open the camera in block 1 first'));
    var total = Number($('settle-s').value) + 3;
    var hz = Number($('tone-hz').value);
    var series = [];
    var prev = null;
    var t0 = performance.now();
    var toneStarted = false;
    toneOff();
    var timer = setInterval(function () {
      var t = (performance.now() - t0) / 1000;
      if (!toneStarted && t >= 3) { toneOn(hz); toneStarted = true; }
      $('settle-clock').textContent = (t < 3 ? 'still… ' : 'tone on · ') + t.toFixed(1) + ' s';
      var g = gray(grabCanvas(160));
      if (prev) {
        var s = 0;
        for (var i = 0; i < g.length; i++) s += Math.abs(g[i] - prev[i]);
        series.push([round(t, 2), round(s / g.length, 4)]);
      }
      prev = g;
      plot(series, 3);
      if (t >= total) {
        clearInterval(timer);
        toneOff();
        $('settle-clock').textContent = '';
        var noise = series.filter(function (p) { return p[0] < 3; }).map(function (p) { return p[1]; }).sort(function (a, b) { return a - b; });
        var floor = noise.length ? noise[Math.floor(noise.length / 2)] : 0;
        var after = series.filter(function (p) { return p[0] >= 3; });
        var peakIdx = 0;
        after.forEach(function (p, i) { if (p[1] > after[peakIdx][1]) peakIdx = i; });
        var settledAt = null;
        for (var k = peakIdx; k + 2 < after.length; k++) {
          if (after[k][1] < 2 * floor && after[k + 1][1] < 2 * floor && after[k + 2][1] < 2 * floor) {
            settledAt = round(after[k][0] - 3, 2);
            break;
          }
        }
        var r = {
          tone_hz: hz,
          sample_period_s: 0.25,
          noise_floor_median: floor,
          peak_change: after.length ? after[peakIdx][1] : null,
          peak_at_s_after_tone: after.length ? round(after[peakIdx][0] - 3, 2) : null,
          settled_at_s_after_tone: settledAt,
          settle_rule: 'three consecutive samples below 2x the median noise floor, after the peak',
          series: series
        };
        report.formation = report.formation || [];
        report.formation.push(r);
        show('b6-out', Object.assign({}, r, { series: '(' + series.length + ' samples, in report)' }));
      }
    }, 250);
  };

  // ------------------------------------------------------------------ 7 save
  $('save').onclick = function () {
    report.notes = $('notes').value;
    report.saved = new Date().toISOString();
    fetch('/probe/report', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(report)
    }).then(json).then(function (r) {
      show('report', report);
      $('report').textContent = 'Saved to ' + r.path + '\n\n' + $('report').textContent;
    }).catch(function (e) { fail('report', e); });
  };
})();

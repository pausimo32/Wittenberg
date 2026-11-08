/**
 * presentation.js
 * Wittenberg — the animated slides of the didactic session ("Ones i so"), played locally.
 *
 * Six slides on a fixed 1920 × 1080 stage, scaled to fit. Every word drawn on a canvas goes
 * through WI18n.t, so the language switcher reaches the animations too. Nothing is fetched:
 * the app runs offline (docs/adr/0005).
 */
(function () {
  'use strict';

  var MA = Math, PI = MA.PI, P2 = 2 * PI;
  var FD = "'Helvetica Neue',Helvetica,Arial,sans-serif";
  var FS = "'Satoshi','Helvetica Neue',Arial,sans-serif";
  var FM = 'ui-monospace,SFMono-Regular,Menlo,Consolas,monospace';

  // The frozen class list (js/classes.generated.js) names the high mode, so the animation
  // quotes the same number the analyser does. The fallbacks only cover a page without it.
  var CLASSES = window.WITTENBERG_CLASSES || [];
  var HZ_LO = CLASSES.length ? CLASSES[0].hz : 165;
  var HZ_HI = CLASSES.length ? CLASSES[CLASSES.length - 1].hz : 2241;

  function t(key) { return window.WI18n ? window.WI18n.t(key) : key; }
  function lang() { return window.WI18n ? window.WI18n.getLang() : 'fr'; }

  // ── Shared drawing helpers ───────────────────────────────────────────────
  function cl(x) { return x > 1 ? 1 : (x < 0 ? 0 : x); }
  function easeOut(u) { return 1 - MA.pow(1 - u, 3); }
  function easeInOut(u) { return u >= 0.5 ? 1 - MA.pow(2 - 2 * u, 3) / 2 : 4 * u * u * u; }
  function fmt(n, k) { return n.toFixed(k).replace('.', ','); }

  function rr(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  function text(g, s, x, y, font, col, align, spacing) {
    g.font = font; g.fillStyle = col; g.textAlign = align || 'left';
    var ls = 'letterSpacing' in g;
    if (ls) g.letterSpacing = (spacing || 0) + 'px';
    g.fillText(s, x, y);
    if (ls) g.letterSpacing = '0px';
  }

  function wrap(g, s, font, maxWidth) {
    g.font = font;
    var words = String(s).split(' '), out = [], cur = '', i, trial;
    for (i = 0; i < words.length; i++) {
      trial = cur ? cur + ' ' + words[i] : words[i];
      if (cur && g.measureText(trial).width > maxWidth) { out.push(cur); cur = words[i]; }
      else cur = trial;
    }
    out.push(cur);
    return out;
  }

  // ── 1. L'ona: a pulse travels, the gold particle only goes up and down ──
  function drawWave(g, T, dt, W, H) {
    g.clearRect(0, 0, W, H);
    var N = 34, HI = 19, m = 40, y0 = H * 0.66, A = H * 0.44, sp = (W - 2 * m) / (N - 1);
    var L = W + 700, p = ((T * 430) % L) - 350, ys = [], i, xi;
    for (i = 0; i < N; i++) { xi = m + i * sp; ys.push(-A * MA.exp(-MA.pow((xi - p) / 120, 2))); }
    g.strokeStyle = '#9DB9C3'; g.lineWidth = 3; g.beginPath();
    for (i = 0; i < N; i++) { xi = m + i * sp; if (i) g.lineTo(xi, y0 + ys[i]); else g.moveTo(xi, y0 + ys[i]); }
    g.stroke();
    var hx = m + HI * sp;
    g.setLineDash([6, 8]); g.strokeStyle = '#CFA96A'; g.lineWidth = 3;
    g.beginPath(); g.moveTo(hx, y0 - A - 4); g.lineTo(hx, y0 + 30); g.stroke(); g.setLineDash([]);
    for (i = 0; i < N; i++) {
      xi = m + i * sp; var hit = i === HI;
      g.fillStyle = hit ? '#CFA96A' : '#205f75';
      g.beginPath(); g.arc(xi, y0 + ys[i], hit ? 16 : 11, 0, 7); g.fill();
      if (hit) { g.strokeStyle = '#123e4d'; g.lineWidth = 3; g.stroke(); }
    }
    var ax = p - 40, ay = y0 - A - 34, len = 110;
    g.strokeStyle = '#205f75'; g.fillStyle = '#205f75'; g.lineWidth = 5;
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(ax + len, ay); g.stroke();
    g.beginPath(); g.moveTo(ax + len + 18, ay); g.lineTo(ax + len - 2, ay - 12); g.lineTo(ax + len - 2, ay + 12);
    g.closePath(); g.fill();
  }

  // ── 2. El so: a loudspeaker pushes and pulls the air ────────────────────
  function drawSound(g, T, dt, W, H) {
    g.clearRect(0, 0, W, H);
    var w = P2 * 0.6, lam = 300, k = P2 / lam, A = 11, x0 = 190, sp = 24, rows = 9, top = 50;
    var rsp = (H - 100) / (rows - 1), cols = MA.floor((W - x0 - 20) / sp), HC = 24, HR = 4, r, i;
    var mem = x0 - 40 + 14 * MA.sin(w * T);
    g.fillStyle = '#123e4d';
    g.beginPath(); g.moveTo(20, H / 2 - 70); g.lineTo(80, H / 2 - 70); g.lineTo(80, H / 2 + 70); g.lineTo(20, H / 2 + 70);
    g.closePath(); g.fill();
    g.beginPath(); g.moveTo(80, H / 2 - 40); g.lineTo(mem, H / 2 - 150); g.lineTo(mem, H / 2 + 150); g.lineTo(80, H / 2 + 40);
    g.closePath(); g.fill();
    g.strokeStyle = '#CFA96A'; g.lineWidth = 6;
    g.beginPath(); g.moveTo(mem, H / 2 - 150); g.lineTo(mem, H / 2 + 150); g.stroke();
    for (r = 0; r < rows; r++) {
      for (i = 0; i < cols; i++) {
        var bx = x0 + i * sp, x = bx + A * MA.sin(w * T - k * (bx - x0)), y = top + r * rsp;
        var hit = i === HC && r === HR;
        g.fillStyle = hit ? '#CFA96A' : '#205f75';
        g.beginPath(); g.arc(x, y, hit ? 11 : 7, 0, 7); g.fill();
        if (hit) { g.strokeStyle = '#123e4d'; g.lineWidth = 3; g.stroke(); }
      }
    }
  }

  // ── 3. Amplitud i freqüència: an automatic demo, or the participant's sliders ──
  var inter = { auto: true, A: 1.4, F: 1.5, ph: 0, amp: null, freq: null, btn: null };

  function oscillationsLabel(F) {
    // French takes the singular below 2; Catalan and Spanish only for exactly one.
    var one = lang() === 'fr' ? F < 2 : F === 1;
    return t(one ? 'pres.inter.osc_one' : 'pres.inter.osc_many').replace('{n}', fmt(F, F % 1 ? 2 : 0));
  }

  function chip(g, x, label, val, frac, on) {
    var FN = FD;
    rr(g, x, 8, 360, 88, 14); g.fillStyle = on ? '#205f75' : '#DCE9ED'; g.fill();
    g.fillStyle = on ? '#F4F8F9' : '#102F3A'; g.font = '26px ' + FN; g.textAlign = 'left'; g.fillText(label, x + 24, 46);
    g.font = 'bold 26px ' + FN; g.textAlign = 'right'; g.fillText(val, x + 336, 46);
    rr(g, x + 24, 66, 312, 8, 4); g.fillStyle = on ? 'rgba(244,248,249,0.3)' : 'rgba(18,62,77,0.18)'; g.fill();
    rr(g, x + 24, 66, MA.max(8, 312 * frac), 8, 4); g.fillStyle = '#CFA96A'; g.fill();
  }

  function drawInteractive(g, T, dt, W, H) {
    var stage = -1, i, v, tt;
    if (inter.auto) {
      tt = T % 18;
      if (tt < 9) { stage = 0; inter.F = 1.5; inter.A = MA.round((1.2 - 0.8 * MA.cos(P2 * tt / 9)) * 10) / 10; }
      else { stage = 1; inter.A = 1.4; inter.F = MA.round((2.25 - 1.75 * MA.cos(P2 * (tt - 9) / 9)) * 4) / 4; }
      syncInterSliders();
    }
    var A = inter.A, F = inter.F;
    inter.ph = (inter.ph + P2 * F * dt) % P2;
    g.clearRect(0, 0, W, H);
    chip(g, 0, t('pres.inter.amp'), fmt(A, 1) + ' ' + t('pres.inter.unit'), A / 2, stage === 0);
    chip(g, 384, t('pres.inter.freq'), fmt(F, 2) + ' Hz', (F - 0.25) / 3.75, stage === 1);
    g.fillStyle = '#4a6370'; g.font = '26px ' + FD; g.textAlign = 'right';
    g.fillText(oscillationsLabel(F), W - 4, 62);
    var L = 100, R = 24, top = 136, bottom = H - 70;
    function X(q) { return L + (W - L - R) * q / 2; }
    function Y(q) { return top + (bottom - top) * (2.2 - q) / 4.4; }
    g.fillStyle = '#FBFDFD'; g.fillRect(L, top, W - L - R, bottom - top);
    g.strokeStyle = 'rgba(18,62,77,0.16)'; g.lineWidth = 1; g.strokeRect(L, top, W - L - R, bottom - top);
    g.font = '24px ' + FD;
    for (v = -2; v <= 2; v++) {
      g.strokeStyle = v === 0 ? '#5F7780' : 'rgba(18,62,77,0.16)'; g.lineWidth = v === 0 ? 2 : 1;
      g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke();
      g.fillStyle = '#4a6370'; g.textAlign = 'right'; g.fillText(String(v).replace('-', '−'), L - 16, Y(v) + 8);
    }
    for (i = 0; i <= 4; i++) {
      var s = i / 2;
      g.strokeStyle = 'rgba(18,62,77,0.12)'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(X(s), top); g.lineTo(X(s), bottom); g.stroke();
      g.fillStyle = '#4a6370'; g.textAlign = i === 0 ? 'left' : (i === 4 ? 'right' : 'center');
      g.fillText(fmt(s, s % 1 ? 1 : 0), X(s), bottom + 32);
    }
    g.fillStyle = '#102F3A'; g.textAlign = 'center'; g.fillText(t('pres.inter.time'), (L + W - R) / 2, H - 6);
    g.save(); g.translate(30, (top + bottom) / 2); g.rotate(-PI / 2); g.fillText(t('pres.inter.disp'), 0, 0); g.restore();
    g.setLineDash([8, 8]); g.strokeStyle = '#CFA96A'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(L, Y(A)); g.lineTo(W - R, Y(A)); g.moveTo(L, Y(-A)); g.lineTo(W - R, Y(-A)); g.stroke();
    g.setLineDash([]);
    g.strokeStyle = '#205f75'; g.lineWidth = 5; g.lineJoin = 'round'; g.lineCap = 'round'; g.beginPath();
    for (i = 0; i <= 700; i++) {
      var q = 2 * i / 700, yy = Y(A * MA.sin(P2 * F * q + inter.ph));
      if (i) g.lineTo(X(q), yy); else g.moveTo(X(q), yy);
    }
    g.stroke();
    g.fillStyle = '#CFA96A'; g.beginPath(); g.arc(X(2), Y(A * MA.sin(P2 * F * 2 + inter.ph)), 11, 0, 7); g.fill();
    g.strokeStyle = '#123e4d'; g.lineWidth = 3; g.stroke();
  }

  function syncInterSliders() {
    if (inter.amp && +inter.amp.value !== inter.A) inter.amp.value = inter.A;
    if (inter.freq && +inter.freq.value !== inter.F) inter.freq.value = inter.F;
    if (inter.btn) inter.btn.setAttribute('aria-pressed', inter.auto ? 'true' : 'false');
  }

  // ── 4. Com es forma una ona estacionària: a 44 s narrated sequence ───────
  var FORM_END = 44.5, FORM_HOLD = 5;

  function drawFormation(g, T) {
    if (T > FORM_END) T = FORM_END;
    var C = { p: '#205f75', d: '#123e4d', bg: '#F4F8F9', b2: '#DCE9ED', a: '#CFA96A', t: '#102F3A', m: '#5F7780' };
    var W0 = 1920, H0 = 1080;
    function en(T, a, b) { return easeOut(cl((T - a) / (b - a))); }
    function dr(T, a, b) { return easeInOut(cl((T - a) / (b - a))); }
    function ex(T, a, b) { var u = cl((T - a) / (b - a)); return 1 - u * u * u; }
    var CU = { Pols: 4, Incident: 11.5, Reflexio: 16.5, Superposicio: 21.5, Nodes: 30.5, Resum: 38.5 };
    var X0 = 260, X1 = 1660, L = 1400, A = 68, K = 3 * PI / L, V = L / 4, WV = K * V, N = 240;
    var LANE1 = 290, LANE2 = 470, REAL = 740, CEN = 540, PW = 300, PH = 120, VP = L / 3.4;
    function pulse(u) { return MA.abs(u) < PW / 2 ? PH * MA.pow(MA.cos(PI * u / PW), 2) : 0; }
    function path(fn, yc, x0, x1) {
      if (x0 === undefined) { x0 = X0; x1 = X1; }
      g.beginPath();
      for (var i = 0; i <= N; i++) { var x = x0 + (x1 - x0) * i / N, y = yc - fn(x - X0); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    }
    function st(col, w, dash, cap) {
      g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.lineCap = cap || 'butt'; g.lineJoin = 'round';
      g.stroke(); g.setLineDash([]);
    }
    function ln(x1, y1, x2, y2, col, w, dash, cap) { g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); st(col, w, dash, cap); }
    function pl(p, col, w) { g.beginPath(); g.moveTo(p[0], p[1]); for (var i = 2; i < p.length; i += 2) g.lineTo(p[i], p[i + 1]); st(col, w); }
    function circ(x, y, r, fill, sc, sw) {
      g.beginPath(); g.arc(x, y, r, 0, 7);
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (sc) { g.strokeStyle = sc; g.lineWidth = sw; g.stroke(); }
    }
    function post(x, yc) { g.fillStyle = C.d; rr(g, x - 7, yc - 110, 14, 220, 3); g.fill(); circ(x, yc, 7, C.bg, C.d, 3); }
    function probe(x, base, y, col) { ln(x, base, x, base - y, col, 7, null, 'round'); circ(x, base - y, 9, col, '#fff', 2.5); }

    var SECS = ['Pols', 'Incident', 'Reflexio', 'Superposicio', 'Nodes', 'Resum'];
    var TI = CU.Pols + L / VP;
    var CAP = [[CU.Pols + 0.4], [TI - 0.6], [TI + 1.4, CU.Incident], [CU.Incident + 0.4], [CU.Reflexio + 0.3],
      [CU.Superposicio + 0.3], [CU.Superposicio + 3], [CU.Superposicio + 5.8], [CU.Nodes + 1], [CU.Nodes + 3],
      [CU.Nodes + 5.4, CU.Resum]];
    var i, j, o, e, a, b, lines, base;

    g.globalAlpha = 1; g.textBaseline = 'alphabetic';
    g.fillStyle = C.bg; g.fillRect(0, 0, W0, H0);
    var c = -PW / 2 + VP * MA.max(0, T - CU.Pols), r = 2 * L - c, pf = ex(T, CU.Pols + 6.6, CU.Pols + 7.3);
    var yP = function (x) { return pf * (pulse(x - c) - pulse(x - r)); };
    var ringP = en(T, TI - 0.2, TI + 1.2), ringO = en(T, TI - 0.3, TI) * ex(T, TI + 0.6, TI + 1.4);
    var crestO = MA.min(en(T, CU.Pols + 0.6, CU.Pols + 1.2), c < L - PW * 0.6 ? 1 : 0) * (T < TI ? 1 : 0);
    var troughO = (T > TI ? 1 : 0) * (r < L - PW * 0.6 ? 1 : 0) * en(T, TI + 0.9, TI + 1.5) * pf;
    var tw = MA.max(0, T - CU.Incident), sf = CU.Superposicio - CU.Incident, tph = tw <= sf ? tw : sf + 0.45 * (tw - sf);
    var fr = V * tph, refl = fr > L;
    var y1 = function (x) { return x <= fr ? A * MA.sin(K * x - WV * tph) : 0; };
    var y2 = function (x) { return x >= 2 * L - fr ? A * MA.sin(K * x + WV * tph) : 0; };
    var ys = function (x) { return y1(x) + y2(x); };
    var yc = CEN + (REAL - CEN) * dr(T, CU.Incident - 0.3, CU.Incident + 0.7) - (REAL - CEN) * dr(T, CU.Nodes, CU.Nodes + 1);
    var realFn = T < CU.Incident ? yP : ys;
    var titleO = ex(T, CU.Pols - 0.6, CU.Pols);
    var laneO = MA.min(en(T, CU.Incident + 0.2, CU.Incident + 1), ex(T, CU.Nodes - 0.2, CU.Nodes + 0.6));
    var lane2O = laneO * en(T, CU.Reflexio - 0.2, CU.Reflexio + 0.4);
    var ovO = MA.min(en(T, CU.Reflexio, CU.Reflexio + 0.6), ex(T, CU.Superposicio + 0.4, CU.Superposicio + 1.2));
    var ovX = X0 + MA.max(0, 2 * L - fr);
    var nPO = MA.min(en(T, CU.Superposicio + 2.8, CU.Superposicio + 3.4), ex(T, CU.Nodes - 0.2, CU.Nodes + 0.4));
    var aPO = MA.min(en(T, CU.Superposicio + 5.6, CU.Superposicio + 6.2), ex(T, CU.Nodes - 0.2, CU.Nodes + 0.4));
    var envO = MA.min(en(T, CU.Nodes + 0.8, CU.Nodes + 1.6), ex(T, CU.Resum, CU.Resum + 0.6));
    var ventO = MA.min(en(T, CU.Nodes + 3, CU.Nodes + 3.8), ex(T, CU.Resum, CU.Resum + 0.5));
    var lamO = MA.min(en(T, CU.Nodes + 5.4, CU.Nodes + 6.2), ex(T, CU.Resum, CU.Resum + 0.5));
    var resO = en(T, CU.Resum + 0.2, CU.Resum + 1);
    var fixO = MA.min(en(T, TI - 0.8, TI - 0.2), ex(T, TI + 1.8, TI + 2.4));

    // Title card
    if (titleO > 0.001) {
      g.globalAlpha = titleO * en(T, 0.2, 0.9);
      ln(120, 164, 160, 164, C.a, 2); text(g, t('pres.form.kicker'), 178, 173, '500 24px ' + FM, C.p, 'left', 3.8);
      e = en(T, 0.4, 1.3); g.globalAlpha = titleO * e;
      lines = wrap(g, t('pres.form.title'), '700 92px ' + FD, 1500);
      for (i = 0; i < lines.length; i++) text(g, lines[i], 120, 283 + i * 97 + (1 - e) * 24, '700 92px ' + FD, C.d, 'left', -1.4);
      g.globalAlpha = titleO * en(T, 1.2, 2);
      lines = wrap(g, t('pres.form.sub'), '400 36px ' + FS, 1300);
      for (i = 0; i < lines.length; i++) text(g, lines[i], 120, 797 + i * 50.4, '400 36px ' + FS, C.m);
    }
    // Section header
    for (i = 0; i < SECS.length; i++) {
      a = CU[SECS[i]]; b = i + 1 < SECS.length ? CU[SECS[i + 1]] : 1e9;
      e = en(T, a, a + 0.6); o = MA.min(e, ex(T, b - 0.35, b));
      if (o > 0.001) {
        g.globalAlpha = o; var dy = (1 - e) * 18;
        ln(120, 82 + dy, 156, 82 + dy, C.a, 2);
        text(g, t('pres.form.header') + ' · 0' + (i + 1) + ' / 06', 174, 90 + dy, '500 22px ' + FM, C.p, 'left', 3.1);
        text(g, t('pres.form.sec' + (i + 1)), 120, 160 + dy, '700 60px ' + FD, C.d, 'left', -0.6);
      }
    }
    // Lanes: the wave that goes and the wave that comes back
    if (laneO > 0.001) {
      g.globalAlpha = laneO;
      ln(X0, LANE1, X1, LANE1, C.b2, 2, [6, 8]); path(y1, LANE1); st(C.p, 5, null, 'round');
      text(g, t('pres.form.lane1'), X1 - 30, LANE1 - 82, '500 24px ' + FM, C.p, 'right', 2.4);
      ln(X0, LANE2, X1, LANE2, C.b2, 2, [6, 8]);
      if (refl) { path(y2, LANE2, ovX, X1); st(C.m, 5, [14, 10], 'round'); }
      g.globalAlpha = lane2O;
      text(g, t('pres.form.lane2'), X1 - 30, LANE2 - 82, '500 24px ' + FM, C.m, 'right', 2.4);
    }
    if (refl) { g.globalAlpha = 0.7 * ovO; g.fillStyle = C.b2; rr(g, ovX, yc - 150, X1 - ovX, 300, 8); g.fill(); }
    o = MA.min(laneO, ex(T, CU.Nodes - 0.5, CU.Nodes));
    if (o > 0.001) { g.globalAlpha = o; text(g, t('pres.form.real'), X1 - 30, REAL - 2 * A - 24, '500 24px ' + FM, C.d, 'right', 2.4); }
    // The real string
    base = 1 - 0.88 * resO; g.globalAlpha = base;
    ln(X0, yc, X1, yc, C.b2, 2);
    if (envO > 0.001) {
      g.globalAlpha = base * envO * 0.55;
      path(function (x) { return 2 * A * MA.sin(K * x); }, yc); st(C.p, 2.5, [10, 9]);
      path(function (x) { return -2 * A * MA.sin(K * x); }, yc); st(C.p, 2.5, [10, 9]);
    }
    g.globalAlpha = base; path(realFn, yc); st(C.d, 8, null, 'round');
    if (crestO > 0.001) { g.globalAlpha = base * crestO; text(g, t('pres.form.crest'), X0 + c, yc - PH - 34, '500 28px ' + FS, C.p, 'center'); }
    if (troughO > 0.001) { g.globalAlpha = base * troughO; text(g, t('pres.form.trough'), X0 + r, yc + PH + 56, '500 28px ' + FS, C.p, 'center'); }
    if (ringO > 0.001) { g.globalAlpha = base * ringO; circ(X1, yc, 16 + 70 * ringP, null, C.a, 5); }
    if (fixO > 0.001) { g.globalAlpha = base * fixO; text(g, t('pres.form.fixed'), X1, yc + 150, '500 24px ' + FM, C.d, 'center', 1.9); }
    if (ventO > 0.001) {
      g.globalAlpha = base * ventO;
      var VX = [L / 6, L / 2, 5 * L / 6];
      for (i = 0; i < 3; i++) {
        var vx = X0 + VX[i];
        ln(vx, yc - 2 * A + 6, vx, yc + 2 * A - 6, C.p, 3);
        pl([vx - 9, yc - 2 * A + 16, vx, yc - 2 * A + 4, vx + 9, yc - 2 * A + 16], C.p, 3);
        pl([vx - 9, yc + 2 * A - 16, vx, yc + 2 * A - 4, vx + 9, yc + 2 * A - 16], C.p, 3);
        text(g, t('pres.form.antinode'), vx, yc + 2 * A + 48, '500 26px ' + FS, C.p, 'center');
      }
    }
    if (lamO > 0.001) {
      g.globalAlpha = base * lamO;
      pl([X0 + L / 3, yc - 196, X0 + L / 3, yc - 176, X0 + 2 * L / 3, yc - 176, X0 + 2 * L / 3, yc - 196], C.d, 3);
      text(g, 'λ / 2', X0 + L / 2, yc - 196, '700 40px ' + FD, C.d, 'center');
    }
    if (envO > 0.001) {
      g.globalAlpha = base * envO;
      for (i = 0; i < 4; i++) {
        circ(X0 + i * L / 3, yc, 13, C.a, C.d, 2.5);
        if (i % 3) text(g, t('pres.form.node'), X0 + i * L / 3, yc - 34, '500 24px ' + FM, C.d, 'center', 1.9);
      }
    }
    g.globalAlpha = base; post(X0, yc); post(X1, yc);
    // Probes: where the two waves cancel, and where they add up
    var PR = [[2 * L / 3, nPO, t('pres.form.cancel')], [L / 6, aPO, t('pres.form.reinforce')]];
    for (i = 0; i < 2; i++) {
      var px = PR[i][0]; o = PR[i][1];
      if (o > 0.001) {
        g.globalAlpha = o; var X = X0 + px;
        ln(X, LANE1 - 90, X, yc + 150, C.a, 2.5, [4, 7]);
        probe(X, LANE1, y1(px), C.p); probe(X, LANE2, y2(px), C.m); probe(X, yc, ys(px), C.d);
        g.font = '700 28px ' + FS;
        rr(g, X + 22, yc + 108, g.measureText(PR[i][2]).width + 36, 50, 8);
        g.fillStyle = '#fff'; g.fill(); g.strokeStyle = C.a; g.lineWidth = 2; g.stroke();
        text(g, PR[i][2], X + 40, yc + 142, '700 28px ' + FS, C.d);
      }
    }
    // Summary cards
    if (T > CU.Resum) {
      var cw = (1680 - 64) / 3, ml = 0, LL = [];
      for (j = 0; j < 3; j++) { LL.push(wrap(g, t('pres.form.res' + (j + 1)), '400 30px ' + FS, cw - 80)); ml = MA.max(ml, LL[j].length); }
      var ch = 40 + 29 + 18 + 58 + 18 + ml * 42 + 44;
      for (j = 0; j < 3; j++) {
        o = en(T, CU.Resum + 0.5 + j * 0.5, CU.Resum + 1.3 + j * 0.5);
        if (o > 0.001) {
          g.globalAlpha = o; var cx = 120 + j * (cw + 32), cy = 280 + (1 - o) * 20;
          g.save(); g.shadowColor = 'rgba(18,62,77,0.08)'; g.shadowBlur = 16; g.shadowOffsetY = 4;
          rr(g, cx, cy, cw, ch, 20); g.fillStyle = '#fff'; g.fill(); g.restore();
          rr(g, cx, cy, cw, ch, 20); g.strokeStyle = 'rgba(18,62,77,0.16)'; g.lineWidth = 1; g.stroke();
          text(g, '0' + (j + 1), cx + 40, cy + 62, '500 24px ' + FM, C.p, 'left', 2.9);
          text(g, t('pres.form.res' + (j + 1) + '_t'), cx + 40, cy + 40 + 29 + 18 + 48, '700 48px ' + FD, C.d);
          for (i = 0; i < LL[j].length; i++) text(g, LL[j][i], cx + 40, cy + 40 + 29 + 18 + 58 + 18 + 30 + i * 42, '400 30px ' + FS, C.t);
        }
      }
      o = en(T, CU.Resum + 2.4, CU.Resum + 3.2);
      if (o > 0.001) {
        g.globalAlpha = o; lines = wrap(g, t('pres.form.chladni'), '500 34px ' + FS, 1616);
        ln(120, 760 + lines.length * 23, 160, 760 + lines.length * 23, C.a, 2);
        for (i = 0; i < lines.length; i++) text(g, lines[i], 184, 793 + i * 45.9, '500 34px ' + FS, C.d);
      }
    }
    // Captions
    var cur = -1;
    for (i = 0; i < CAP.length; i++) if (T >= CAP[i][0]) cur = i;
    if (cur >= 0 && (CAP[cur][1] === undefined || T < CAP[cur][1])) {
      o = en(T, CAP[cur][0], CAP[cur][0] + 0.35);
      if (CAP[cur][1] !== undefined) o = MA.min(o, ex(T, CAP[cur][1] - 0.3, CAP[cur][1]));
      g.globalAlpha = o; lines = wrap(g, t('pres.form.cap' + (cur + 1)), '500 34px ' + FS, 1459);
      var capTop = 1080 - 44 - lines.length * 45.9;
      for (i = 0; i < lines.length; i++) text(g, lines[i], 960, capTop + 34 + i * 45.9, '500 34px ' + FS, C.d, 'center');
    }
    g.globalAlpha = 1;
  }

  // ── 5. Ona estacionària: nodes stay still, antinodes move the most ──────
  function drawStanding(g, T, dt, W, H) {
    g.clearRect(0, 0, W, H);
    var m = 70, L = W - 2 * m, y0 = H / 2, A = H * 0.38, cw = MA.cos(P2 * 0.6 * T), x, j, n = 3, step = 4;
    g.fillStyle = '#123e4d'; g.fillRect(m - 30, y0 - 110, 30, 220); g.fillRect(W - m, y0 - 110, 30, 220);
    g.setLineDash([8, 10]); g.strokeStyle = '#9DB9C3'; g.lineWidth = 2;
    [-1, 1].forEach(function (sgn) {
      g.beginPath();
      for (x = 0; x <= L; x += step) { var e = sgn * A * MA.sin(n * PI * x / L); if (x) g.lineTo(m + x, y0 - e); else g.moveTo(m + x, y0 - e); }
      g.stroke();
    });
    g.setLineDash([]);
    g.strokeStyle = '#205f75'; g.lineWidth = 7; g.lineCap = 'round'; g.beginPath();
    for (x = 0; x <= L; x += step) { var yy = y0 - A * MA.sin(n * PI * x / L) * cw; if (x) g.lineTo(m + x, yy); else g.moveTo(m + x, yy); }
    g.stroke();
    for (j = 0; j < n; j++) {
      var ax = (j + 0.5) * L / n, ay = y0 - A * MA.sin(n * PI * ax / L) * cw;
      g.fillStyle = '#205f75'; g.beginPath(); g.arc(m + ax, ay, 15, 0, 7); g.fill();
      g.strokeStyle = '#F4F8F9'; g.lineWidth = 4; g.stroke();
    }
    for (j = 0; j <= n; j++) {
      g.fillStyle = '#CFA96A'; g.beginPath(); g.arc(m + j * L / n, y0, 17, 0, 7); g.fill();
      g.strokeStyle = '#123e4d'; g.lineWidth = 4; g.stroke();
    }
  }

  // ── 6. Les línies nodals: the plate, the salt, the sweep and the model ──
  // Salt grain positions traced from the real plate at the high mode, 2 base64 digits per
  // coordinate (12 bits, 0..4095). Copied from the session deck, not regenerated.
  var NODAL_DATA = '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';
  var nodal = null;

  function rng(seed) {
    var s = seed >>> 0;
    return function () { s = (MA.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  }
  function modeHi(x, y) {
    var s8 = MA.pow(MA.sin(PI * x), 8), h = 0.3625 + 0.6625 * MA.cos(4 * PI * y) - 0.525 * MA.cos(2 * PI * y);
    return 0.5 * (MA.cos(6 * PI * x) + s8 * h + 0.6 * MA.cos(4 * PI * x) * MA.cos(2 * PI * y));
  }
  function modeLo(x, y) { return 0.6 * (MA.cos(2 * PI * x) - 0.2 + 0.229 * MA.cos(2 * PI * y)); }

  function buildNodal() {
    var NG = 720, SC = [], PH = [], TH = [], GS = [], r1 = rng(7), r2 = rng(31), r3 = rng(71), r4 = rng(53), i, j;
    for (i = 0; i < NG; i++) {
      SC.push([0.08 + r1() * 0.84, 0.08 + r1() * 0.84]); PH.push(r2() * P2); TH.push((r3() - 0.5) * 2); GS.push(1.05 + r4() * 1.15);
    }
    var AL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    function dq(k) { return (AL.indexOf(NODAL_DATA.charAt(k)) * 64 + AL.indexOf(NODAL_DATA.charAt(k + 1))) / 4095; }
    var T0 = [];
    for (i = 0; i < NG; i++) T0.push([dq(i * 4), dq(i * 4 + 2)]);
    // The low mode: zero crossings of modeLo on a grid, then one grain per crossing.
    var G = 460, pts = [], v = new Float64Array((G + 1) * (G + 1)), lo = 0.035, hi = 0.965, T2 = [];
    for (i = 0; i <= G; i++) for (j = 0; j <= G; j++) v[i * (G + 1) + j] = modeLo(i / G, j / G);
    function inside(x, y) { return x > lo && x < hi && y > lo && y < hi; }
    for (i = 0; i < G; i++) {
      for (j = 0; j < G; j++) {
        var a = v[i * (G + 1) + j], bx = v[(i + 1) * (G + 1) + j], by = v[i * (G + 1) + j + 1], tt, x, y;
        if ((a < 0) !== (bx < 0)) { tt = a / (a - bx); x = (i + tt) / G; y = j / G; if (inside(x, y)) pts.push([x, y]); }
        if ((a < 0) !== (by < 0)) { tt = a / (a - by); x = i / G; y = (j + tt) / G; if (inside(x, y)) pts.push([x, y]); }
      }
    }
    var stp = pts.length / NG, r = rng(39);
    for (i = 0; i < NG; i++) {
      var p = pts[MA.min(pts.length - 1, MA.floor((i + r() * 0.9) * stp))];
      var gx = 0.6 * (-P2 * MA.sin(P2 * p[0])), gy = 0.6 * (-0.229 * P2 * MA.sin(P2 * p[1])), gn = MA.hypot(gx, gy) || 1, off = TH[i] * 0.0075;
      T2.push([p[0] + gx / gn * off, p[1] + gy / gn * off]);
    }
    return { NG: NG, SC: SC, PH: PH, GS: GS, T0: T0, T2: T2 };
  }

  function drawNodal(g, T) {
    if (!nodal) nodal = buildNodal();
    var NG = nodal.NG, SC = nodal.SC, PH = nodal.PH, GS = nodal.GS, T0 = nodal.T0, T2 = nodal.T2;
    var C = { bgD: '#082430', p: '#205f75', pl: '#3f92ad', d: '#123e4d', l: '#F4F8F9', s: '#DCE9ED', a: '#CFA96A', m: '#7fa3b0' };
    var U = { Y: 0, P: 5, S: 11, E: 18, M: 24, D: 30 }, TOT = 35;
    function A(v) { g.globalAlpha = v; }
    function lin(u) { return u; }
    function an(T, f, to, a, b, e) { return f + (to - f) * e(cl((T - a) / (b - a))); }
    function gl(T, F, e) {
      e = e || easeInOut;
      if (T <= F[0]) return F[1];
      for (var i = 0; i < F.length - 2; i += 2) if (T <= F[i + 2]) return an(T, F[i + 1], F[i + 3], F[i], F[i + 2], e);
      return F[F.length - 1];
    }
    function en(T, s, d) { return an(T, 0, 1, s, s + (d || 0.6), easeOut); }
    function fd(T, a, b, c, d) { return MA.min(en(T, a, b - a), 1 - en(T, c, d - c)); }
    function prj(u, v, w, cam) {
      var cy = MA.cos(cam.yaw), sy = MA.sin(cam.yaw), xr = u * cy - v * sy, yr = u * sy + v * cy;
      return [cam.cx + xr * cam.sc, cam.cy + (yr * MA.cos(cam.pi) - w * MA.sin(cam.pi)) * cam.sc];
    }
    function vib(T) {
      var f0 = 0.4, f1 = 3.1, a = 2.2, b = 18.5, k = (f1 - f0) / (b - a);
      if (T <= a) return P2 * f0 * T;
      var d = MA.min(T, b) - a, ph = f0 * a + f0 * d + 0.5 * k * d * d;
      if (T > b) ph += f1 * (T - b);
      return P2 * ph;
    }
    function tw(s, font) { g.font = font; return g.measureText(s).width; }
    function ln(pts, col, w, close) {
      g.beginPath(); g.moveTo(pts[0], pts[1]);
      for (var k = 2; k < pts.length; k += 2) g.lineTo(pts[k], pts[k + 1]);
      if (close) g.closePath();
      g.strokeStyle = col; g.lineWidth = w; g.stroke();
    }
    function box(x, y, w, h, r, fill, sc, sw) {
      rr(g, x, y, w, h, r);
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (sc) { g.strokeStyle = sc; g.lineWidth = sw; g.stroke(); }
    }
    function dot(x, y, r, col) { g.beginPath(); g.arc(x, y, r, 0, 7); g.fillStyle = col; g.fill(); }
    function note(x, y, X, Y, title, sub, col, op, al) {
      if (op < 0.01) return;
      A(op * 0.75); ln([x, y, X, Y], col, 1.6); A(op); dot(x, y, 4.5, col);
      text(g, title, X, Y - 8, '700 27px ' + FD, col, al, 0.6);
      if (sub) text(g, sub, X, Y + 20, '21px ' + FS, C.m, al);
    }

    g.lineCap = 'butt'; g.lineJoin = 'round'; A(1);
    var cam = {
      pi: gl(T, [0, 1.525, U.P + 0.2, 1.525, U.P + 2.4, 0.95, U.S - 0.2, 0.95, U.S + 2.2, 0.19, U.E + 5.2, 0.19, U.M + 1.4, 0.06, TOT, 0.06]),
      yaw: gl(T, [0, 0, U.P + 0.2, 0, U.P + 2.4, 0.42, U.S - 0.2, 0.42, U.S + 2.2, 0, TOT, 0]),
      sc: gl(T, [0, 640, U.P, 640, U.P + 2.4, 720, U.S + 2.2, 780, U.E, 660, U.M - 0.6, 660, U.M + 1.4, 320, U.D + 0.7, 320, U.D + 1.9, 470, TOT, 470]),
      cx: gl(T, [0, 700, U.P + 2.4, 900, U.S + 2.2, 900, U.E, 900, U.M - 0.6, 900, U.M + 1.4, 750, U.D + 0.7, 750, U.D + 1.9, 560, TOT, 560]),
      cy: gl(T, [0, 640, U.P + 2.4, 560, U.S + 2.2, 470, U.M - 0.6, 470, U.M + 1.4, 470, U.D + 0.7, 470, U.D + 1.9, 560, TOT, 560])
    };
    var amp = 0.075 * gl(T, [0, 0, 2.5, 0, 3.4, 1, U.E + 0.6, 1, U.E + 1.2, 0.28, U.E + 3.4, 0.28, U.E + 4.1, 1, U.M + 0.3, 1, U.M + 1.3, 0, TOT, 0]);
    var ph = vib(T), osc = MA.sin(ph), blend = gl(T, [U.E + 1.2, 0, U.E + 4.1, 1]), m2 = blend >= 0.5;
    var Wf = m2 ? modeLo : modeHi, TG = m2 ? T2 : T0;
    var lock = gl(T, [U.S + 0.4, 0, U.S + 3.6, 1, U.E + 1.2, 1, U.E + 1.7, 0.12, U.E + 3.5, 0.12, U.E + 4.4, 1, TOT, 1]);
    var jig = gl(T, [U.E + 1.2, 0, U.E + 1.7, 1, U.E + 3.6, 1, U.E + 4.2, 0]);
    var heatOp = fd(T, U.P + 0.6, U.P + 1.6, U.M + 0.3, U.M + 1.2) * 0.62 + 0.05;
    var wireOp = fd(T, U.P + 1.4, U.P + 2.6, U.M + 0.4, U.M + 1.4), edgeOp = fd(T, U.P + 1.4, U.P + 2.6, TOT - 0.6, TOT - 0.1);
    var hz = MA.round(gl(T, [0, HZ_HI, U.E + 1, HZ_HI, U.E + 4, HZ_LO, TOT, HZ_LO], lin));
    var vig = fd(T, 0, 0.6, TOT - 0.7, TOT - 0.05), k, i, j, x, y, P, pts;

    // The gradient stops 54 px short of the bottom edge; the base fill covers that strip.
    g.fillStyle = C.bgD; g.fillRect(0, 0, 1920, 1080);
    g.save(); g.translate(960, 486); g.scale(1920, 1080);
    var gr = g.createRadialGradient(0, 0, 0, 0, 0, 0.7);
    gr.addColorStop(0, '#12475b'); gr.addColorStop(1, C.bgD);
    g.fillStyle = gr; g.fillRect(-0.5, -0.45, 1, 1); g.restore();
    A(0.95); g.fillStyle = C.a; g.fillRect(72, 64, 16, 16); text(g, 'WITTENBERG', 102, 80, '700 25px ' + FD, C.l, 'left', 4.2);
    A(fd(T, 0.6, 1.4, TOT - 0.9, TOT - 0.2));
    text(g, hz + ' Hz', 1848, 88, '700 64px ' + FD, C.a, 'right');
    text(g, t('pres.lin.hz_label'), 1848, 120, '21px ' + FS, C.m, 'right', 1.6);
    var fo = fd(T, U.P + 1.4, U.P + 2.4, TOT - 0.9, TOT - 0.2);
    if (fo > 0.01) {
      A(fo);
      var FP = ['w(x,y) = cos(', 'm', 'πx/L)·cos(', 'n', 'πy/L) ± cos(', 'n', 'πx/L)·cos(', 'm', 'πy/L)'], fx = 72;
      for (k = 0; k < FP.length; k++) {
        var f = (k % 2 ? '700 ' : '500 ') + '29px ' + FD;
        text(g, FP[k], fx, 1016, f, k % 2 ? C.a : C.s); fx += tw(FP[k], f);
      }
    }
    var bo = fd(T, U.P + 2.2, U.P + 3.2, TOT - 0.9, TOT - 0.2) * (1 - fd(T, U.E - 0.9, U.E - 0.3, U.M - 0.4, U.M + 0.4));
    if (bo > 0.01) {
      A(bo); text(g, t('pres.lin.formula_head'), 1848, 938, '18px ' + FS, C.m, 'right', 2.6);
      var BR = [['w(x,y)', 'pres.lin.f1'], ['m, n', 'pres.lin.f2'], ['±', 'pres.lin.f3']];
      for (k = 0; k < 3; k++) {
        var line = '  ' + t(BR[k][1]);
        text(g, line, 1848, 976 + k * 32, '21px ' + FS, C.s, 'right');
        text(g, BR[k][0], 1848 - tw(line, '21px ' + FS), 976 + k * 32, '700 21px ' + FD, C.a, 'right');
      }
    }
    var so = fd(T, 0.2, 0.9, U.P + 0.9, U.P + 1.9), tX = cam.cx, tY = cam.cy + 46;
    if (so > 0.01) {
      A(so); box(90, 830, 230, 122, 10, C.d, C.p, 2);
      text(g, t('pres.lin.generator'), 112, 858, '19px ' + FS, C.m, 'left', 1.4);
      pts = []; for (k = 0; k <= 58; k++) pts.push(112 + k * 2.6, 884 + MA.sin(k * 0.44 - ph * 1.9) * 14);
      ln(pts, C.a, 2.4);
      text(g, HZ_HI + ' Hz', 112, 938, '700 26px ' + FD, C.l); ln([320, 891, 470, 891], C.p, 3);
      if (T > 1.05 && T < 2.3) dot(320 + cl((T - 1.05) / 1.15) * 150, 891, 6, C.a);
      box(470, 839, 150, 104, 8, C.d, C.p, 2);
      text(g, 'AMPLI', 545, 879, '17px ' + FS, C.m, 'center', 1.2); text(g, '▲', 545, 917, '700 24px ' + FD, C.l, 'center');
      ln([620, 891, tX, 891], C.p, 3); A(so * 0.85);
      g.beginPath(); g.moveTo(tX - 40, 891); g.lineTo(tX + 40, 891); g.lineTo(tX + 20, 806); g.lineTo(tX - 20, 806); g.closePath();
      g.fillStyle = C.p; g.fill();
      A(so); ln([tX, 806, tX, tY + 6], C.a, 5); dot(tX, tY + 6, 7, C.a);
      var up = cl((T - 2.2) / 0.6);
      if (up > 0) { A(so * (1 - up * 0.6)); dot(tX, 806 - (806 - tY) * up, 6, C.l); }
      A(so); text(g, t('pres.lin.transducer'), tX + 62, 874, '19px ' + FS, C.m);
    }
    var po = fd(T, 0.3, 1.1, U.P + 0.9, U.P + 1.7);
    if (po > 0.01) {
      A(po); g.beginPath(); var top = [];
      for (k = 0; k <= 60; k++) { x = k / 60; top.push(prj(x - 0.5, 0, Wf(x, 0.5) * amp * osc, cam)); }
      g.moveTo(top[0][0], top[0][1] - 4.5);
      for (k = 1; k < 61; k++) g.lineTo(top[k][0], top[k][1] - 4.5);
      for (k = 60; k >= 0; k--) g.lineTo(top[k][0], top[k][1] + 4.5);
      g.closePath(); g.fillStyle = C.s; g.fill(); g.strokeStyle = C.l; g.lineWidth = 1; g.stroke();
    }
    function edge() {
      g.beginPath(); var CO = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0];
      for (var q = 0; q < 4; q++) {
        for (var s = 0; s <= 24; s++) {
          var xx = CO[q * 2] + (CO[q * 2 + 2] - CO[q * 2]) * s / 24, yy = CO[q * 2 + 1] + (CO[q * 2 + 3] - CO[q * 2 + 1]) * s / 24;
          var E = prj(xx - 0.5, yy - 0.5, Wf(xx, yy) * amp * osc, cam);
          if (q + s) g.lineTo(E[0], E[1]); else g.moveTo(E[0], E[1]);
        }
      }
    }
    if (edgeOp > 0.01) { A(edgeOp * 0.62); edge(); g.fillStyle = '#0e3b4c'; g.fill(); }
    if (wireOp > 0.01) {
      A(wireOp * 0.5); g.strokeStyle = C.pl; g.lineWidth = 1;
      for (var dd = 0; dd < 2; dd++) {
        for (i = 0; i < 13; i++) {
          g.beginPath();
          for (j = 0; j < 33; j++) {
            var s1 = j / 32, c1 = i / 12; x = dd ? c1 : s1; y = dd ? s1 : c1;
            P = prj(x - 0.5, y - 0.5, Wf(x, y) * amp * osc, cam);
            if (j) g.lineTo(P[0], P[1]); else g.moveTo(P[0], P[1]);
          }
          g.stroke();
        }
      }
    }
    if (heatOp > 0.01) {
      for (i = 0; i < 15; i++) {
        for (j = 0; j < 15; j++) {
          x = (i + 0.5) / 15; y = (j + 0.5) / 15; var w = Wf(x, y), aa = MA.abs(w);
          P = prj(x - 0.5, y - 0.5, w * amp * osc, cam);
          A(heatOp * (0.28 + aa * 0.72)); dot(P[0], P[1], 2 + aa * 3.4, aa < 0.18 ? C.s : (aa > 0.72 ? C.a : C.pl));
        }
      }
    }
    if (edgeOp > 0.01) { A(edgeOp); edge(); g.strokeStyle = C.s; g.lineWidth = 2.2; g.stroke(); }
    var sOp = en(T, U.S - 0.6, 1) * fd(T, U.S - 0.6, U.S + 0.4, TOT - 0.6, TOT - 0.1);
    if (sOp > 0.01) {
      g.fillStyle = C.l; var wob = (1 - lock) * jig;
      for (i = 0; i < NG; i++) {
        x = SC[i][0] + (TG[i][0] - SC[i][0]) * lock; y = SC[i][1] + (TG[i][1] - SC[i][1]) * lock;
        x += MA.sin(ph * 0.9 + PH[i]) * 0.018 * wob; y += MA.cos(ph * 1.1 + PH[i] * 1.7) * 0.018 * wob;
        var a2 = MA.abs(Wf(x, y)), hop = a2 * amp * (0.55 + 0.45 * MA.sin(ph * 2 + PH[i] * 3)) * 1.5;
        P = prj(x - 0.5, y - 0.5, hop, cam); A(sOp * (0.72 + a2 * 0.28));
        g.beginPath(); g.arc(P[0], P[1], GS[i], 0, 7); g.fill();
      }
    }
    var eL = prj(-0.5, 0.5, 0, cam), eR = prj(0.5, 0.5, 0, cam), eRv = prj(0.5, 0.5, Wf(1, 1) * amp * osc, cam);
    note(eRv[0], eRv[1], eR[0] + 40, eR[1] - 60, 'e = 1 mm', t('pres.lin.thick_sub'), C.s, fd(T, 0.8, 1.6, U.P + 1, U.P + 1.8));
    var dm = fd(T, 1.4, 2.2, U.P + 1, U.P + 1.8);
    if (dm > 0.01) {
      A(dm); ln([eL[0], eL[1] - 96, eR[0], eR[1] - 96], C.m, 1.6);
      ln([eL[0], eL[1] - 106, eL[0], eL[1] - 86], C.m, 1.6); ln([eR[0], eR[1] - 106, eR[0], eR[1] - 86], C.m, 1.6);
      text(g, '18 cm', (eL[0] + eR[0]) / 2, eR[1] - 110, '23px ' + FS, C.m, 'center');
    }
    note(cam.cx, cam.cy + 20, cam.cx + 320, cam.cy + 160, t('pres.lin.mech'), t('pres.lin.mech_sub'), C.a, fd(T, 2.6, 3.3, U.P + 0.6, U.P + 1.4));
    var AN = [0.3333, 0.5], NO = [0.5354, 0.4239];
    var ap = prj(AN[0] - 0.5, AN[1] - 0.5, modeHi(AN[0], AN[1]) * amp * osc, cam), as = prj(AN[0] - 0.5, AN[1] - 0.5, 0, cam);
    var np = prj(NO[0] - 0.5, NO[1] - 0.5, 0, cam);
    note(ap[0], ap[1], as[0] - 90, as[1] - 120, t('pres.lin.antinode'), t('pres.lin.antinode_sub'), C.a, fd(T, U.P + 2.6, U.P + 3.4, U.S + 3.4, U.S + 4.2), 'right');
    note(np[0], np[1], np[0] + 120, np[1] - 96, t('pres.lin.node'), t('pres.lin.node_sub'), C.s, fd(T, U.P + 3.4, U.P + 4.2, U.S + 3.4, U.S + 4.2));
    var sb = fd(T, U.S + 1.4, U.S + 2.2, U.E - 0.2, U.E + 0.4);
    if (sb > 0.01) {
      // 446 px keeps the box clear of the plate; a longer translated title shrinks instead.
      var SL = wrap(g, t('pres.lin.salt_body'), '22px ' + FS, 394), bh = 112 + (SL.length - 1) * 32;
      var st = t('pres.lin.salt_title'), sz = MA.min(24, MA.floor(24 * 394 / tw(st, '700 24px ' + FD)));
      A(sb * 0.92); box(72, 690, 446, bh, 12, C.d); A(sb); box(72, 690, 446, bh, 12, null, C.p, 1.5);
      text(g, st, 98, 748, '700 ' + sz + 'px ' + FD, C.a, 'left', 0.4);
      for (k = 0; k < SL.length; k++) text(g, SL[k], 98, 790 + k * 32, '22px ' + FS, C.s);
    }
    var sw = fd(T, U.E - 0.4, U.E + 0.5, U.M - 0.5, U.M + 0.2);
    if (sw > 0.01) {
      A(sw); text(g, t('pres.lin.sweep'), 620, 864, '21px ' + FS, C.m, 'left', 1.6); ln([620, 924, 1520, 924], C.p, 3);
      var HZ = [HZ_LO, HZ_HI];
      for (k = 0; k < 2; k++) {
        x = 620 + HZ[k] / 2400 * 900; var on = MA.abs(hz - HZ[k]) < 16;
        ln([x, 914, x, 946], on ? C.a : C.p, on ? 4 : 2); text(g, String(HZ[k]), x, 976, '20px ' + FS, on ? C.a : C.m, 'center');
      }
      x = 620 + cl(hz / 2400) * 900;
      g.beginPath(); g.moveTo(x, 898); g.lineTo(x + 11, 878); g.lineTo(x - 11, 878); g.closePath(); g.fillStyle = C.a; g.fill();
      dot(x, 924, 8, C.a);
      text(g, t(lock > 0.6 ? 'pres.lin.res_on' : 'pres.lin.res_off'), 1520, 864, '700 26px ' + FD, C.s, 'right');
    }
    var mo = fd(T, U.M + 0.6, U.M + 1.6, U.D - 0.1, U.D + 0.6);
    if (mo > 0.01) {
      A(mo); box(1010, 310, 320, 320, 6, C.d, C.p, 1.5); A(mo * 0.9); g.fillStyle = C.l;
      for (i = 0; i < NG; i++) { g.beginPath(); g.arc(1010 + T0[i][0] * 320, 310 + T0[i][1] * 320, 1.5, 0, 7); g.fill(); }
      A(mo);
      text(g, HZ_HI + ' Hz', 1170, 668, '700 30px ' + FD, C.a, 'center'); text(g, t('pres.lin.real_mode'), 1170, 698, '22px ' + FS, C.m, 'center');
      text(g, HZ_LO + ' Hz', 750, 668, '700 30px ' + FD, C.a, 'center'); text(g, t('pres.lin.real_mode'), 750, 698, '22px ' + FS, C.m, 'center');
      ln([928, 470, 992, 470], C.a, 3); ln([976, 454, 992, 470, 976, 486], C.a, 3);
      text(g, t('pres.lin.cmp_title'), 960, 232, '700 34px ' + FD, C.l, 'center');
      text(g, t('pres.lin.cmp_sub'), 960, 272, '22px ' + FS, C.m, 'center');
    }
    var co = en(T, U.D + 2.1, 0.8);
    if (co > 0.01) {
      A(co * 0.95); box(1120, 330, 700, 420, 16, C.d); A(co); box(1120, 330, 700, 420, 16, null, C.p, 1.5);
      text(g, t('pres.lin.cnn_head'), 1164, 392, '21px ' + FS, C.m, 'left', 1.6);
      box(1164, 420, 160, 160, 8, C.bgD, C.p, 1.5); A(co * 0.9); g.fillStyle = C.l;
      for (i = 0; i < NG; i += 2) { g.beginPath(); g.arc(1164 + T2[i][0] * 160, 420 + T2[i][1] * 160, 1.3, 0, 7); g.fill(); }
      for (k = 0; k < 3; k++) { A(co * (0.4 + k * 0.2)); var hh = 120 - k * 26; box(1372 + k * 54, 500 - hh / 2, 30, hh, 4, C.p); }
      A(co); text(g, t('pres.lin.cnn_layers'), 1372, 596, '20px ' + FS, C.m);
      ln([1580, 490, 1640, 490], C.a, 2.5); ln([1626, 476, 1640, 490, 1626, 504], C.a, 2.5);
      text(g, t('pres.lin.cnn_pred'), 1664, 470, '20px ' + FS, C.m, 'left', 1.2);
      text(g, HZ_LO + ' Hz', 1664, 516, '700 46px ' + FD, C.a);
      var CB = wrap(g, t('pres.lin.cnn_body'), '23px ' + FS, 612);
      for (k = 0; k < CB.length; k++) text(g, CB[k], 1164, 668 + k * 32, '23px ' + FS, C.s);
    }
    A(1 - vig); g.fillStyle = C.bgD; g.fillRect(0, 0, 1920, 1080); A(1);
  }

  // ── Player ───────────────────────────────────────────────────────────────
  // `period` loops the clock; `poster` is the frame shown when motion is reduced, chosen
  // so a still image still teaches something.
  var SLIDES = [
    { id: 'ona',   canvas: 'cv-ona',   w: 1664, h: 420,  poster: 1.6, draw: drawWave },
    { id: 'so',    canvas: 'cv-so',    w: 1664, h: 420,  poster: 0.6, draw: drawSound },
    { id: 'inter', canvas: 'cv-inter', w: 1664, h: 470,  poster: 0,   draw: drawInteractive },
    { id: 'form',  canvas: 'cv-form',  w: 1920, h: 1080, poster: 37,  draw: drawFormation, period: FORM_END + FORM_HOLD },
    { id: 'estac', canvas: 'cv-estac', w: 1664, h: 400,  poster: 0,   draw: drawStanding },
    { id: 'lin',   canvas: 'cv-lin',   w: 1920, h: 1080, poster: 15,  draw: drawNodal, period: 35 }
  ];

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var state = { index: 0, playing: !reduceMotion, elapsed: 0, last: null, scale: 1, dirty: true };
  var els = {};

  function stageFit() {
    var vw = els.viewport.clientWidth, vh = els.viewport.clientHeight;
    var s = MA.min(vw / 1920, vh / 1080), ox = (vw - 1920 * s) / 2, oy = (vh - 1080 * s) / 2;
    els.stage.style.transform = 'translate(' + ox + 'px,' + oy + 'px) scale(' + s + ')';
    state.scale = s;
    SLIDES.forEach(sizeCanvas);
    state.dirty = true;
  }

  function sizeCanvas(slide) {
    var cv = slide.el, k = (window.devicePixelRatio || 1) * state.scale;
    var bw = MA.max(1, MA.round(slide.w * k)), bh = MA.max(1, MA.round(slide.h * k));
    if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
    slide.k = bw / slide.w;
  }

  function render(dt) {
    var slide = SLIDES[state.index], g = slide.ctx;
    var T = slide.period ? state.elapsed % slide.period : state.elapsed;
    g.setTransform(slide.k, 0, 0, slide.k, 0, 0);
    g.globalAlpha = 1;
    slide.draw(g, T, dt, slide.w, slide.h);
  }

  function frame(now) {
    var dt = state.last === null ? 0 : MA.min(0.1, (now - state.last) / 1000);
    state.last = now;
    if (state.playing) { state.elapsed += dt; render(dt); }
    else if (state.dirty) render(0);
    state.dirty = false;
    requestAnimationFrame(frame);
  }

  function go(index, fromHash) {
    var n = SLIDES.length;
    index = (index + n) % n;
    state.index = index;
    state.elapsed = state.playing ? 0 : SLIDES[index].poster;
    state.dirty = true;
    SLIDES.forEach(function (s, i) {
      var on = i === index;
      s.section.classList.toggle('is-active', on);
      s.section.setAttribute('aria-hidden', on ? 'false' : 'true');
      if ('inert' in s.section) s.section.inert = !on;
    });
    els.dots.forEach(function (d, i) { d.setAttribute('aria-current', i === index ? 'step' : 'false'); });
    if (!fromHash && history.replaceState) history.replaceState(null, '', '#' + SLIDES[index].id);
    updateText();
  }

  function setPlaying(on) {
    state.playing = on;
    state.dirty = true;
    updateText();
  }

  function restart() {
    state.elapsed = 0;
    inter.ph = 0;
    state.dirty = true;
  }

  function isFullscreen() { return (document.fullscreenElement || document.webkitFullscreenElement) === els.player; }

  function toggleFullscreen() {
    if (isFullscreen()) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    } else {
      var req = els.player.requestFullscreen || els.player.webkitRequestFullscreen;
      if (req) req.call(els.player);
    }
  }

  function updateText() {
    var slide = SLIDES[state.index], n = SLIDES.length;
    els.counter.textContent = (state.index + 1) + ' / ' + n;
    els.noteTitle.textContent = t('pres.' + slide.id + '.title');
    els.noteStep.textContent = t('pres.notes_step').replace('{i}', state.index + 1).replace('{n}', n);
    els.note.textContent = t('pres.' + slide.id + '.note');
    els.play.querySelector('.pres-btn-icon').textContent = state.playing ? '❚❚' : '▶';
    els.play.querySelector('.pres-btn-label').textContent = t(state.playing ? 'pres.pause' : 'pres.play');
    els.play.setAttribute('aria-pressed', state.playing ? 'true' : 'false');
    els.fs.querySelector('.pres-btn-label').textContent = t(isFullscreen() ? 'pres.exit_fullscreen' : 'pres.fullscreen');
    els.dots.forEach(function (d, i) {
      d.setAttribute('aria-label', t('pres.goto').replace('{i}', i + 1) + ' — ' + t('pres.' + SLIDES[i].id + '.title'));
    });
    SLIDES.forEach(function (s, i) {
      s.section.setAttribute('aria-label', (i + 1) + ' / ' + n + ' — ' + t('pres.' + s.id + '.title'));
    });
  }

  function bindInteractive() {
    inter.amp = document.getElementById('inter-amp');
    inter.freq = document.getElementById('inter-freq');
    inter.btn = document.getElementById('inter-auto');
    function manual() {
      inter.auto = false;
      inter.A = +inter.amp.value;
      inter.F = +inter.freq.value;
      syncInterSliders();
      state.dirty = true;
    }
    inter.amp.addEventListener('input', manual);
    inter.freq.addEventListener('input', manual);
    inter.btn.addEventListener('click', function () {
      inter.auto = !inter.auto;
      if (inter.auto) state.elapsed = 0;
      syncInterSliders();
      state.dirty = true;
    });
    syncInterSliders();
  }

  function init() {
    els.player = document.getElementById('pres-player');
    if (!els.player) return;
    els.viewport = document.getElementById('pres-viewport');
    els.stage = document.getElementById('pres-stage');
    els.counter = document.getElementById('pres-counter');
    els.note = document.getElementById('pres-note');
    els.noteTitle = document.getElementById('pres-note-title');
    els.noteStep = document.getElementById('pres-note-step');
    els.play = document.getElementById('pres-play');
    els.fs = document.getElementById('pres-fs');

    var dotsBox = document.getElementById('pres-dots');
    els.dots = SLIDES.map(function (s, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'pres-dot';
      b.addEventListener('click', function () { go(i); });
      dotsBox.appendChild(b);
      return b;
    });

    SLIDES.forEach(function (s) {
      s.el = document.getElementById(s.canvas);
      s.ctx = s.el.getContext('2d');
      s.section = document.querySelector('[data-slide="' + s.id + '"]');
    });

    document.getElementById('pres-prev').addEventListener('click', function () { go(state.index - 1); });
    document.getElementById('pres-next').addEventListener('click', function () { go(state.index + 1); });
    document.getElementById('pres-restart').addEventListener('click', restart);
    els.play.addEventListener('click', function () { setPlaying(!state.playing); });
    els.fs.addEventListener('click', toggleFullscreen);
    if (!(els.player.requestFullscreen || els.player.webkitRequestFullscreen)) els.fs.hidden = true;
    bindInteractive();

    document.addEventListener('keydown', function (e) {
      var tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); go(state.index + 1); }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(state.index - 1); }
      else if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); setPlaying(!state.playing); }
      else if (e.key === 'f' || e.key === 'F') { toggleFullscreen(); }
    });
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
      document.addEventListener(ev, function () { stageFit(); updateText(); });
    });
    document.addEventListener('wittenberg:langchange', function () { updateText(); state.dirty = true; });
    if (window.ResizeObserver) new ResizeObserver(stageFit).observe(els.viewport);
    else window.addEventListener('resize', stageFit);
    // Canvas text uses the page's fonts; redraw once they have loaded, or a paused slide
    // would keep the fallback face.
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { state.dirty = true; });
    document.getElementById('pres-reduced').hidden = !reduceMotion;

    var fromHash = -1;
    SLIDES.forEach(function (s, i) { if (location.hash === '#' + s.id) fromHash = i; });
    window.addEventListener('hashchange', function () {
      SLIDES.forEach(function (s, i) { if (location.hash === '#' + s.id && i !== state.index) go(i, true); });
    });

    stageFit();
    go(fromHash >= 0 ? fromHash : 0, true);
    requestAnimationFrame(function () { els.stage.classList.add('is-ready'); });
    requestAnimationFrame(frame);
  }

  document.addEventListener('DOMContentLoaded', init);
})();

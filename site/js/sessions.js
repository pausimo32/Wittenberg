/**
 * sessions.js
 * Wittenberg — capture sessions, `docs/05-webapp-spec.md` §9, on the static site.
 *
 * A port of the React wizard in `web/src/routes/dades/` (ADR 0007: `web/` is ported from,
 * then removed). The behaviour is that wizard's, and so are the rules it carried:
 *
 * - **A draft is server state.** `drafts/<session_id>/` on disk, not this tab's memory, so a
 *   reload or a second tab lands on the same afternoon. This page holds only what the server
 *   has not been asked yet, and re-reads the draft after anything that changes it.
 * - **The order is the server's.** `POST /frames` answers 409 `not_calibrated` until four
 *   markers are found; the photo step is shut here too, so nobody discovers the order by
 *   selecting sixty photographs.
 * - **Bytes leave untouched.** A `File` from the picker goes into the `FormData` as it is —
 *   no canvas, no re-encoding. `data/raw/` is defined as straight off the phone (`docs/02` §4).
 * - **Verdicts stream.** Three uploads at a time, one verdict per file, and a refusal is a
 *   result: the eleventh photo being unreadable stops nothing.
 * - **Three counters, not two.** Accepted, rejected and duplicated: a repeated upload is not a
 *   lighting problem and must not be counted as one.
 * - **The server's own words when there is no translation.** An unknown gate reason or refusal
 *   is shown as the server's token, never papered over with an invented sentence.
 *
 * One deliberate change from the React version: the classes are offered from
 * `js/classes.generated.js` instead of typed. The class list is frozen now (`config/classes.json`),
 * and a typed label is a label that can be typed wrong.
 */
(function () {
  'use strict';

  var POLL_MS = 500;
  var UPLOAD_CONCURRENCY = 3;
  var ACCEPT = '.heic,.heif,.jpg,.jpeg,.png,image/*';
  var CLASSES = (window.WITTENBERG_CLASSES || []).map(function (c) { return c.hz; });

  function t(key, vars) {
    var s = window.WI18n ? window.WI18n.t(key) : key;
    if (vars) Object.keys(vars).forEach(function (k) { s = s.split('{' + k + '}').join(String(vars[k])); });
    return s;
  }

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function plural(n, one, many, vars) {
    vars = vars || {};
    vars.n = n;
    return t(n === 1 ? one : many, vars);
  }

  // ── API: same origin, relative, no retry (docs/adr/0004) ──────────────────
  function ApiError(message, status, code, info) {
    this.message = message;
    this.status = status;
    this.code = code || '';
    this.info = info || {};
  }

  async function failure(response) {
    var detail = response.statusText, code = '', info = {};
    try {
      var body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
      else if (body.detail && typeof body.detail === 'object') {
        if (typeof body.detail.message === 'string') detail = body.detail.message;
        if (typeof body.detail.code === 'string') code = body.detail.code;
        info = body.detail;
      }
    } catch (e) { /* a non-JSON error body is not itself worth reporting */ }
    return new ApiError(detail, response.status, code, info);
  }

  async function call(url, options) {
    var response;
    try { response = await fetch(url, options); }
    catch (e) { throw new ApiError(t('sess.err.network'), 0); }
    if (!response.ok) throw await failure(response);
    return response.status === 204 ? null : response.json();
  }

  function draftUrl(id) { return '/api/drafts/' + encodeURIComponent(id); }

  function upload(url, file) {
    var form = new FormData();
    form.append('file', file, file.name); // the File itself, never a re-encoded copy
    return call(url, { method: 'POST', body: form });
  }

  var api = {
    list: function () { return call('/api/drafts').then(function (b) { return b.drafts; }); },
    create: function (log) {
      return call('/api/drafts', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(log) });
    },
    get: function (id) { return call(draftUrl(id)); },
    discard: function (id) { return call(draftUrl(id), { method: 'DELETE' }); },
    calibrate: function (id, file) { return upload(draftUrl(id) + '/calibration', file); },
    frame: function (id, hz, file) { return upload(draftUrl(id) + '/frames?freq_hz=' + hz, file); },
    dropFrame: function (id, hz, name) {
      return call(draftUrl(id) + '/frames/' + hz + '/' + encodeURIComponent(name), { method: 'DELETE' });
    },
    commit: function (id, replace) { return call(draftUrl(id) + '/commit?replace=' + Boolean(replace), { method: 'POST' }); },
    commitStatus: function (id) { return call(draftUrl(id) + '/commit'); },
    split: function () { return call('/api/dataset/split', { method: 'POST' }); },
    matrix: function () { return call('/api/dataset/matrix'); },
    preview: function (id, hz, name, kind) {
      return draftUrl(id) + '/preview/' + hz + '/' + encodeURIComponent(name) + '?kind=' + kind;
    }
  };

  async function uploadQueue(items, worker, onResult) {
    var next = 0;
    async function drain() {
      while (next < items.length) {
        var item = items[next++];
        try { onResult(item, await worker(item), null); }
        catch (e) { onResult(item, null, e instanceof ApiError ? e : new ApiError(String(e), 0)); }
      }
    }
    var lanes = [];
    for (var i = 0; i < Math.min(UPLOAD_CONCURRENCY, items.length); i++) lanes.push(drain());
    await Promise.all(lanes);
  }

  // ── State ─────────────────────────────────────────────────────────────────
  var view = document.getElementById('sx-view');
  var serverBox = document.getElementById('sx-server');
  var S = freshState();
  var pollToken = 0;

  function freshState() {
    return {
      route: '', drafts: null, listError: null, confirming: null,
      form: null, formError: null, sending: false,
      draft: null, draftError: null,
      calib: null, calibBusy: false, calibError: null, calibRetry: false,
      seen: {}, commit: null, refusal: null, commitBusy: false,
      split: null, splitRefusal: null, splitting: false,
      matrix: null, matrixError: null
    };
  }

  function today() {
    var d = new Date(), m = String(d.getMonth() + 1).padStart(2, '0'), day = String(d.getDate()).padStart(2, '0');
    return { iso: d.getFullYear() + '-' + m + '-' + day, compact: d.getFullYear() + m + day };
  }

  function emptyForm() {
    var d = today();
    // The rig's own values (docs/02 §1, and the session parameters the team fixed on
    // 2026-11-08): prefilled, not assumed — each one is a field, and varying it is a keystroke.
    return {
      session_id: d.compact + '-01', date: d.iso, operator: '', plate: '',
      material: 'sand', grain_mass_g: '10', distance_mm: '210', lighting: 'lamp_diffuse',
      dwell_s: '5', camera: 'iphone13-heic', notes: ''
    };
  }

  // ── Routing: sessions.html, sessions.html#new, sessions.html#<draft_id> ───
  function route() {
    var hash = decodeURIComponent(location.hash.replace(/^#/, ''));
    pollToken++;
    S = freshState();
    S.route = hash;
    if (!hash) loadList();
    else if (hash === 'new') { S.form = emptyForm(); render(); }
    else loadDraft(true);
  }

  function go(hash) {
    var target = hash ? '#' + encodeURIComponent(hash) : '';
    // Setting the hash it already has fires no `hashchange`, so route by hand.
    if (location.hash === target) route();
    else location.hash = target;
  }

  async function loadList() {
    render();
    try { S.drafts = await api.list(); }
    catch (e) { S.listError = t('sess.err.list'); S.drafts = []; }
    render();
  }

  async function loadDraft(first) {
    if (first) render();
    try { S.draft = await api.get(S.route); S.draftError = null; }
    catch (e) { S.draftError = e.status === 404 ? t('sess.err.not_found') : t('sess.err.load'); }
    render();
    if (first && S.draft) { pollCommit(); loadMatrix(); }
  }

  async function loadMatrix() {
    try { S.matrix = await api.matrix(); S.matrixError = null; }
    catch (e) { S.matrixError = e.message; }
    render();
  }

  // One request on arrival, then one after each answer while it says `running`. Chained,
  // never setInterval, so a slow answer cannot stack a queue behind itself.
  function pollCommit() {
    var token = pollToken, id = S.route;
    (async function poll() {
      var answer;
      try { answer = await api.commitStatus(id); }
      catch (e) { if (token === pollToken) { S.refusal = new ApiError(t('sess.err.network'), 0); render(); } return; }
      if (token !== pollToken) return;
      var wasRunning = S.commit && S.commit.status === 'running';
      S.commit = answer;
      render();
      if (answer.status === 'running') setTimeout(poll, POLL_MS);
      else if (wasRunning && answer.status === 'done') loadMatrix();
    })();
  }

  // ── Server status ─────────────────────────────────────────────────────────
  var serverState = null;
  async function checkServer() {
    try {
      var r = await fetch('/api/health');
      serverState = r.ok ? ((await r.json()).mode === 'capture' ? 'capture' : 'serving') : 'down';
    } catch (e) { serverState = 'down'; }
    renderServer();
  }

  function renderServer() {
    if (serverState === 'down') serverBox.innerHTML = '<p class="sx-warn" style="margin-bottom:1rem;">' + t('sess.server_down') + '</p>';
    else if (serverState === 'capture') serverBox.innerHTML = '<p class="sx-info" style="margin-bottom:1rem;">' + esc(t('sess.capture_mode')) + '</p>';
    else serverBox.innerHTML = '';
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  function render() {
    if (!S.route) view.innerHTML = renderList();
    else if (S.route === 'new') view.innerHTML = renderForm();
    else view.innerHTML = renderDraft();
  }

  function loading() { return '<p role="status" class="sx-muted">' + esc(t('sess.loading')) + '</p>'; }
  function alertBox(html) { return '<div role="alert" class="sx-warn" style="margin-top:1rem;">' + html + '</div>'; }

  function sizeText(bytes) {
    var units = ['B', 'kB', 'MB', 'GB', 'TB'], v = bytes, u = 0;
    while (v >= 1000 && u < units.length - 1) { v /= 1000; u++; }
    return v.toFixed(u === 0 ? 0 : 1).replace('.', ',') + ' ' + units[u];
  }

  function countsText(a, r, d) {
    return d > 0 ? t('sess.drafts.counts_dup', { a: a, r: r, d: d }) : t('sess.drafts.counts', { a: a, r: r });
  }

  function renderList() {
    var out = '<div class="sx-row" style="justify-content:space-between;margin-bottom:1rem;">' +
      '<p class="sx-muted">' + esc(t('sess.drafts.untouched')) + '</p>' +
      '<span class="sx-row"><a class="btn btn-primary" href="acquisition.html">' + esc(t('sess.station_link')) + '</a>' +
      '<button type="button" class="btn btn-outline" data-action="new">' + esc(t('sess.new')) + '</button></span></div>' +
      '<h2 class="sx-h2">' + esc(t('sess.drafts.title')) + '</h2>';
    if (S.listError) out += alertBox(esc(S.listError));
    if (S.drafts === null) return out + loading();
    if (!S.drafts.length) return out + '<p class="sx-muted">' + esc(t('sess.drafts.none')) + '</p>';
    S.drafts.forEach(function (d) {
      out += '<div class="sx-box">' +
        '<div class="sx-row" style="justify-content:space-between;"><h3 class="sx-h3 sx-mono">' + esc(d.draft_id) + '</h3>' +
        '<span class="sx-muted" style="font-size:.8125rem;">' + esc(t('sess.drafts.updated')) + ': ' + esc(String(d.updated).replace('T', ' ')) + '</span></div>' +
        '<p style="margin-top:.5rem;">' + esc(countsText(d.accepted, d.rejected, d.duplicates)) + '</p>' +
        (d.pending > 0 ? '<p class="sx-muted" style="font-size:.875rem;">' + esc(t('sess.drafts.pending', { n: d.pending })) + '</p>' : '') +
        '<p class="sx-muted" style="font-size:.875rem;">' + esc(sizeText(d.bytes)) + '</p>' +
        '<p class="sx-muted" style="font-size:.8125rem;margin-top:.5rem;">' + esc(t('sess.drafts.path')) + '</p>' +
        '<p class="sx-mono" style="font-size:.8125rem;word-break:break-all;">' + esc(d.path) + '</p>' +
        (d.status === 'committing' ? alertBox(esc(t('sess.drafts.interrupted'))) : '') +
        '<div class="sx-row" style="margin-top:1rem;">' +
        '<button type="button" class="btn btn-primary btn-sm" data-action="open" data-id="' + esc(d.draft_id) + '">' + esc(t('sess.drafts.open')) + '</button>' +
        (S.confirming !== d.draft_id ? '<button type="button" class="btn btn-outline btn-sm" data-action="confirm" data-id="' + esc(d.draft_id) + '">' + esc(t('sess.drafts.discard')) + '</button>' : '') +
        '</div>' +
        (S.confirming === d.draft_id
          ? '<div role="alertdialog" class="sx-warn" style="margin-top:1rem;"><p>' + esc(t('sess.drafts.discard_confirm', { id: d.draft_id })) + '</p>' +
            '<div class="sx-row" style="margin-top:.75rem;"><button type="button" class="btn btn-dark btn-sm" data-action="discard" data-id="' + esc(d.draft_id) + '">' + esc(t('sess.drafts.discard')) + '</button>' +
            '<button type="button" class="btn btn-outline btn-sm" data-action="cancel">' + esc(t('sess.cancel')) + '</button></div></div>'
          : '') +
        '</div>';
    });
    return out;
  }

  function field(name, label, input, hint, wide) {
    return '<div class="sx-field' + (wide ? ' sx-wide' : '') + '"><label for="f-' + name + '">' + esc(label) + '</label>' + input +
      (hint ? '<p class="sx-hint" id="f-' + name + '-hint">' + esc(hint) + '</p>' : '') + '</div>';
  }

  function input(name, type, extra) {
    return '<input id="f-' + name + '" data-field="' + name + '" type="' + type + '" value="' + esc(S.form[name]) + '" ' + (extra || '') + '>';
  }

  function select(name, options) {
    return '<select id="f-' + name + '" data-field="' + name + '">' + options.map(function (o) {
      return '<option value="' + o[0] + '"' + (S.form[name] === o[0] ? ' selected' : '') + '>' + esc(t(o[1])) + '</option>';
    }).join('') + '</select>';
  }

  function renderForm() {
    var out = '<p><a href="sessions.html" data-action="home">' + esc(t('sess.back')) + '</a></p>' +
      '<p class="sx-muted" style="margin-top:1rem;">' + esc(t('sess.steps.session')) + '</p>' +
      '<h2 class="sx-h2">' + esc(t('sess.form.title')) + '</h2>' +
      '<p class="sx-muted" style="margin-bottom:1.25rem;">' + esc(t('sess.form.lead')) + '</p>';
    if (S.formError) {
      // The server's own sentence, verbatim, beside its code: it knows why it refused.
      out += alertBox('<p>' + esc(S.formError.message) + '</p>' + (S.formError.code ? '<p class="sx-mono" style="font-size:.8125rem;">' + esc(S.formError.code) + '</p>' : ''));
    }
    out += '<form id="sx-form" class="sx-box sx-form" style="margin-top:1rem;">' +
      field('session_id', t('sess.form.id'), input('session_id', 'text', 'required class="sx-mono" aria-describedby="f-session_id-hint"'), t('sess.form.id_hint')) +
      field('date', t('sess.form.date'), input('date', 'date', 'required')) +
      field('operator', t('sess.form.operator'), input('operator', 'text', 'required')) +
      field('plate', t('sess.form.plate'), input('plate', 'text', 'required'), t('sess.form.plate_hint')) +
      field('material', t('sess.form.material'), select('material', [['sand', 'sess.form.mat_sand'], ['semolina', 'sess.form.mat_semolina'], ['salt', 'sess.form.mat_salt']])) +
      field('grain_mass_g', t('sess.form.grain'), input('grain_mass_g', 'number', 'required step="0.1" min="0"')) +
      field('distance_mm', t('sess.form.distance'), input('distance_mm', 'number', 'required step="1" min="0"')) +
      field('lighting', t('sess.form.lighting'), select('lighting', [['lamp_diffuse', 'sess.form.light_diffuse'], ['lamp_grazing', 'sess.form.light_grazing'], ['ambient', 'sess.form.light_ambient']])) +
      field('dwell_s', t('sess.form.dwell'), input('dwell_s', 'number', 'required step="0.1" min="0"')) +
      field('camera', t('sess.form.camera'), select('camera', [['iphone13-heic', 'sess.form.camera_heic'], ['iphone13-continuity-png', 'sess.form.camera_png']])) +
      // No `required` on notes: empty means "nothing anomalous", which is information (docs/02 §4).
      field('notes', t('sess.form.notes'), '<textarea id="f-notes" data-field="notes" rows="3" aria-describedby="f-notes-hint">' + esc(S.form.notes) + '</textarea>', t('sess.form.notes_hint'), true) +
      '<div class="sx-wide"><button type="submit" class="btn btn-primary"' + (S.sending ? ' disabled aria-busy="true"' : '') + '>' + esc(t('sess.form.submit')) + '</button></div>' +
      '</form>';
    return out;
  }

  async function submitForm() {
    var f = S.form;
    S.sending = true; S.formError = null; render();
    try {
      // docs/02 §4's keys with §4's types: numbers leave as numbers. The session id is not
      // validated here — `metadata.validate_session_id` does it, and its message is shown.
      var draft = await api.create({
        session_id: f.session_id.trim(), date: f.date, operator: f.operator.trim(), plate: f.plate.trim(),
        material: f.material, grain_mass_g: Number(f.grain_mass_g), distance_mm: Number(f.distance_mm),
        lighting: f.lighting, dwell_s: Number(f.dwell_s), camera: f.camera, notes: f.notes
      });
      go(draft.draft_id);
    } catch (e) {
      S.formError = { message: e.status ? e.message : t('sess.err.create'), code: e.code };
      S.sending = false; render();
    }
  }

  // ── The draft: four steps ─────────────────────────────────────────────────
  function currentStep(d) {
    if (!d.calibrated) return 1;
    return d.total > 0 ? 3 : 2;
  }

  function renderDraft() {
    var d = S.draft;
    var out = '<p><a href="sessions.html" data-action="home">' + esc(t('sess.back')) + '</a></p>' +
      '<h2 class="sx-h2 sx-mono" style="margin-top:1rem;">' + esc(S.route) + '</h2>';
    if (d) out += '<p class="sx-muted" style="margin-top:-.5rem;margin-bottom:1rem;">' + esc(t('sess.drafts.status_' + d.status)) + '</p>';
    if (S.draftError && !committed()) {
      out += alertBox('<p>' + esc(S.draftError) + '</p><button type="button" class="btn btn-outline btn-sm" style="margin-top:.5rem;" data-action="reload">' + esc(t('sess.retry')) + '</button>');
    }
    if (!d && !S.draftError) return out + loading();
    if (!d) return out;

    var step = currentStep(d), locked = !d.calibrated;
    var steps = [['session', ''], ['calibration', renderCalibration(d)], ['frames', locked ? '<p class="sx-muted" style="margin-top:.5rem;">' + esc(t('sess.cal.locked')) + '</p>' : renderFrames(d)], ['review', renderReview(d)]];
    out += '<ol class="sx-steps">';
    steps.forEach(function (s, i) {
      out += '<li class="sx-step"' + (i === step ? ' aria-current="step"' : '') + (s[0] === 'frames' && locked ? ' aria-disabled="true"' : '') + '>' +
        '<h3 class="sx-h3">' + esc(t('sess.steps.' + s[0])) + '</h3>' + s[1] + '</li>';
    });
    return out + '</ol>' + renderMatrix();
  }

  function dropZone(target, text, multiple, inputId) {
    return '<div class="sx-drop" data-drop="' + target + '">' +
      '<p>' + esc(text) + '</p>' +
      '<input class="sx-sr" id="' + inputId + '" type="file" accept="' + ACCEPT + '" data-files="' + target + '"' + (multiple ? ' multiple' : '') + '>' +
      '<label for="' + inputId + '" class="btn btn-primary btn-sm" style="margin-top:.75rem;cursor:pointer;">' + esc(t('sess.choose')) + '</label></div>';
  }

  function renderCalibration(d) {
    var calibration = S.calib ? S.calib.calibration : d.calibration;
    var preview = S.calib ? S.calib.preview : null;
    var showDrop = (calibration === null || S.calibRetry) && !S.calibBusy;
    var out = '<p class="sx-muted" style="margin-top:.5rem;">' + esc(t('sess.cal.lead')) + '</p>';
    if (showDrop) out += dropZone('calib', t('sess.cal.drop'), false, 'sx-calib-input');
    if (S.calibBusy) out += '<p role="status" style="margin-top:.75rem;">' + esc(t('sess.cal.working')) + '</p>';
    if (S.calibError) out += alertBox(esc(S.calibError));
    if (calibration) {
      out += '<div class="sx-stack" style="margin-top:1rem;">' +
        '<p>' + esc(t('sess.cal.found', { n: calibration.n_markers })) + '</p>' +
        (calibration.ok
          ? '<p>' + esc(t('sess.cal.ok')) + '</p>' + (calibration.px_per_mm !== null ? '<p class="sx-mono">' + esc(calibration.px_per_mm.toFixed(1)) + ' px/mm</p>' : '')
          : '<p class="sx-warn">' + esc(t('sess.cal.failed')) + '</p>') +
        (preview ? '<img class="sx-preview" src="data:image/png;base64,' + preview + '" alt="' + esc(t('sess.cal.preview_alt')) + '">' : '') +
        // The rig flags are English prose from Python, shown as they are: paraphrasing a rig
        // report would be inventing its second half.
        (calibration.flags.length ? '<div class="sx-warn"><strong>' + esc(t('sess.cal.flags')) + '</strong><ul style="margin-top:.4rem;padding-left:1.2rem;list-style:disc;">' +
          calibration.flags.map(function (f) { return '<li>' + esc(f) + '</li>'; }).join('') + '</ul></div>' : '') +
        (!showDrop ? '<button type="button" class="btn btn-outline btn-sm" data-action="recalibrate">' + esc(t('sess.cal.retry')) + '</button>' : '') +
        '</div>';
    }
    return out;
  }

  async function calibrate(file) {
    S.calibBusy = true; S.calibError = null; render();
    try {
      S.calib = await api.calibrate(S.route, file);
      S.calibRetry = false;
      await loadDraft(false);
    } catch (e) {
      S.calibError = e.message; S.calibRetry = true;
    }
    S.calibBusy = false; render();
  }

  function isRejected(r) { return r.quality !== 'ok' && r.quality !== 'pending'; }

  function reasonText(reason) {
    var key = 'sess.reason.' + reason, s = t(key);
    return s === key ? reason : s;
  }

  function refusalText(e) {
    if (e.code === 'duplicate_frame') return t('sess.frames.duplicate');
    if (e.code === 'duplicate_content') return t('sess.frames.duplicate_content');
    if (e.code === 'unsupported_image') return t('sess.frames.unsupported');
    return e.message;
  }

  function seenOf(hz) {
    return S.seen[hz] || (S.seen[hz] = { records: [], duplicates: 0, problems: [], done: 0, total: 0 });
  }

  /** The server's list for one class, with anything watched land since merged over it by name. */
  function recordsOf(d, hz) {
    var byName = {};
    (d.frames[String(hz)] || []).forEach(function (r) { byName[r.name] = r; });
    seenOf(hz).records.forEach(function (r) { byName[r.name] = r; });
    return Object.keys(byName).sort().map(function (k) { return byName[k]; });
  }

  function counter(label, value) {
    return '<p class="sx-counter"><span>' + esc(label) + '</span><span>' + value + '</span></p>';
  }

  function renderFrames(d) {
    var classes = CLASSES.slice();
    (d.classes || []).forEach(function (hz) { if (classes.indexOf(hz) < 0) classes.push(hz); });
    classes.sort(function (a, b) { return a - b; });
    var out = '<p class="sx-muted" style="margin-top:.5rem;">' + esc(t('sess.frames.lead')) + '</p><div class="sx-classes">';
    classes.forEach(function (hz) {
      var records = recordsOf(d, hz), seen = seenOf(hz);
      var ok = records.filter(function (r) { return r.quality === 'ok'; });
      var rejected = records.filter(isRejected);
      var pending = records.filter(function (r) { return r.quality === 'pending'; }).length;
      out += '<section class="sx-class"><h4 class="sx-h3 sx-mono">' + hz + ' Hz</h4><div class="sx-counters">' +
        counter(t('sess.frames.accepted'), ok.length) + counter(t('sess.frames.rejected'), rejected.length) +
        counter(t('sess.frames.duplicated'), seen.duplicates) +
        (pending ? counter(t('sess.frames.pending'), pending) : '') + '</div>' +
        dropZone('freq:' + hz, t('sess.frames.drop'), true, 'sx-frames-' + hz);
      if (seen.done < seen.total) {
        out += '<p role="status" style="margin-top:.5rem;"><span class="sx-mono">' + esc(t('sess.frames.uploading', { done: seen.done, total: seen.total })) +
          '</span> <span class="sx-muted">' + esc(t('sess.frames.keep_going')) + '</span></p>';
      }
      if (seen.problems.length) {
        out += '<ul class="sx-rejects">' + seen.problems.map(function (p) {
          return '<li class="sx-reject"><span class="sx-mono">' + esc(p.filename) + '</span><br>' + esc(p.message) + '</li>';
        }).join('') + '</ul>';
      }
      if (rejected.length) {
        out += '<ul class="sx-rejects">' + rejected.map(function (r) {
          // No four markers, no rectified plate, no edge map: the server writes none.
          var edges = r.reason !== 'partial';
          return '<li class="sx-reject"><p class="sx-mono">' + esc(r.name) + '</p><p>' + esc(r.original_filename) + '</p>' +
            (r.reason ? '<p style="font-weight:600;">' + esc(reasonText(r.reason)) + '</p>' : '') +
            (r.preview ? '<div class="sx-thumbs"><img src="' + esc(api.preview(d.draft_id, hz, r.name, 'original')) + '" alt="' + esc(t('sess.frames.original')) + '">' +
              (edges ? '<img src="' + esc(api.preview(d.draft_id, hz, r.name, 'edges')) + '" alt="' + esc(t('sess.frames.edges')) + '">' : '') + '</div>' : '') +
            '<button type="button" class="sx-small-btn" style="margin-top:.4rem;" data-action="drop-frame" data-hz="' + hz + '" data-name="' + esc(r.name) + '">' + esc(t('sess.frames.remove')) + '</button></li>';
        }).join('') + '</ul>';
      }
      if (ok.length) {
        out += '<ul class="sx-oklist">' + ok.map(function (r) {
          return '<li><span class="sx-mono">' + esc(r.name) + '</span><span class="sx-muted">' + esc(r.original_filename) + '</span>' +
            '<button type="button" class="sx-small-btn" data-action="drop-frame" data-hz="' + hz + '" data-name="' + esc(r.name) + '">' + esc(t('sess.frames.remove')) + '</button></li>';
        }).join('') + '</ul>';
      }
      out += '</section>';
    });
    return out + '</div>';
  }

  async function uploadFrames(hz, fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) return;
    var seen = seenOf(hz), id = S.route, token = pollToken;
    seen.total += files.length;
    render();
    await uploadQueue(files, function (file) { return api.frame(id, hz, file); }, function (file, record, error) {
      if (token !== pollToken) return;
      seen.done++;
      if (record) seen.records.push(record);
      else {
        if (error.code === 'duplicate_content') seen.duplicates++;
        seen.problems.push({ filename: file.name, message: refusalText(error) });
      }
      render();
    });
    // Once, at the end of the batch: the counters have been moving all along.
    if (token === pollToken) loadDraft(false);
  }

  async function dropFrame(hz, name) {
    try {
      await api.dropFrame(S.route, hz, name);
      var seen = seenOf(hz);
      seen.records = seen.records.filter(function (r) { return r.name !== name; });
      await loadDraft(false);
    } catch (e) { S.draftError = e.message; render(); }
  }

  // ── Review and commit ─────────────────────────────────────────────────────
  function committed() { return S.commit && S.commit.status === 'done'; }

  function renderReview(d) {
    var records = [], reasons = {}, byClass = [];
    Object.keys(d.frames).forEach(function (hz) {
      var entries = d.frames[hz];
      records = records.concat(entries);
      byClass.push({ hz: Number(hz), ok: entries.filter(function (r) { return r.quality === 'ok'; }).length, total: entries.length });
    });
    byClass.sort(function (a, b) { return a.hz - b.hz; });
    records.filter(isRejected).forEach(function (r) { var k = r.reason || r.quality; reasons[k] = (reasons[k] || 0) + 1; });
    var accepted = records.filter(function (r) { return r.quality === 'ok'; }).length, total = records.length;
    var running = S.commit && S.commit.status === 'running';

    var out = '<p class="sx-muted" style="margin-top:.5rem;">' + esc(t('sess.review.lead')) + '</p>' +
      '<div class="sx-row" style="gap:2rem;margin-top:.75rem;">' +
      counter(t('sess.review.total'), total) +
      counter(t('sess.review.yield'), accepted + ' / ' + total + (total ? ' · ' + Math.round(accepted / total * 100) + ' %' : '')) + '</div>';
    if (d.duplicates > 0) out += '<p style="margin-top:.5rem;">' + esc(plural(d.duplicates, 'sess.review.dup_one', 'sess.review.dup_many')) + '</p>';
    if (byClass.length) {
      out += '<p style="font-weight:600;margin-top:1rem;">' + esc(t('sess.review.by_class')) + '</p><div class="sx-row" style="gap:1.5rem;">' +
        byClass.map(function (c) { return counter(c.hz + ' Hz', c.ok + '/' + c.total); }).join('') + '</div>';
    }
    var reasonKeys = Object.keys(reasons).sort(function (a, b) { return reasons[b] - reasons[a]; });
    if (reasonKeys.length) {
      out += '<p style="font-weight:600;margin-top:1rem;">' + esc(t('sess.frames.rejected')) + '</p><ul>' +
        reasonKeys.map(function (k) { return '<li><span class="sx-mono">' + reasons[k] + '</span> ' + esc(reasonText(k)) + '</li>'; }).join('') + '</ul>';
    }
    if (S.commit && S.commit.status === 'interrupted') out += alertBox(esc(t('sess.drafts.interrupted')));
    if (S.commit && S.commit.status === 'failed') {
      out += alertBox('<p style="font-weight:600;">' + esc(t('sess.review.failed')) + '</p>' + (S.commit.error ? '<p class="sx-mono" style="font-size:.8125rem;">' + esc(S.commit.error.message) + '</p>' : ''));
    }
    if (S.refusal) {
      if (S.refusal.code === 'session_exists') {
        // The override is a second, explicit press — never an automatic retry with replace=true.
        out += alertBox('<p>' + esc(t('sess.review.exists', { id: d.session_id, rows: Number(S.refusal.info.existing_rows) })) + '</p>' +
          '<button type="button" class="btn btn-outline btn-sm" style="margin-top:.5rem;" data-action="commit-replace"' + (S.commitBusy || running ? ' disabled' : '') + '>' + esc(t('sess.review.replace')) + '</button>');
      } else if (S.refusal.code === 'mixed_provenance') {
        out += alertBox(esc(t('sess.review.mixed')));
      } else {
        out += alertBox(esc(S.refusal.message));
      }
    }
    if (running) out += '<p role="status" class="sx-mono" style="margin-top:1rem;font-size:1.125rem;">' + esc(progressText(S.commit)) + '</p>';
    if (!committed()) {
      out += '<button type="button" class="btn btn-primary" style="margin-top:1rem;" data-action="commit"' + (S.commitBusy || running || total === 0 ? ' disabled' : '') + '>' + esc(t('sess.review.commit')) + '</button>';
    } else {
      var rep = S.commit.report;
      out += '<div class="sx-stack" style="margin-top:1rem;"><p style="font-weight:600;font-size:1.125rem;">' + esc(t('sess.review.done')) + '</p>' +
        (rep ? '<p class="sx-mono">' + rep.accepted + '/' + rep.total + '</p>' : '') +
        '<p class="sx-warn">' + esc(t('sess.review.relabel')) + '</p>' +
        '<button type="button" class="btn btn-primary" data-action="split"' + (S.splitting ? ' disabled' : '') + '>' + esc(t('sess.review.split')) + '</button>' +
        (S.split ? '<p role="status" class="sx-mono">' + esc(t('sess.review.split_done', { train: S.split.images.train || 0, val: S.split.images.val || 0, test: S.split.images.test || 0 })) + '</p>' : '') +
        (S.splitRefusal ? alertBox(esc(S.splitRefusal.code === 'split_failed' ? t('sess.review.split_failed') : S.splitRefusal.message)) : '') +
        '</div>';
    }
    return out;
  }

  function progressText(status) {
    if (status.phase === 'stage') return t('sess.review.staging');
    if (status.phase === 'rename') return t('sess.review.renaming');
    return t('sess.review.committing', { done: status.done, total: status.total });
  }

  async function commit(replace) {
    S.refusal = null; S.commitBusy = true; render();
    try { S.commit = await api.commit(S.route, replace); pollToken++; pollCommit(); }
    catch (e) { S.refusal = e; }
    S.commitBusy = false; render();
  }

  async function resplit() {
    S.splitRefusal = null; S.splitting = true; render();
    try { S.split = await api.split(); }
    catch (e) { S.splitRefusal = e; }
    S.splitting = false; render(); loadMatrix();
  }

  // An empty cell is drawn empty, never 0/0: a zero is a measurement, a blank is no session.
  function renderMatrix() {
    var m = S.matrix;
    var out = '<section class="sx-box" style="margin-top:2rem;"><h3 class="sx-h3">' + esc(t('sess.matrix.title')) + '</h3>' +
      '<p class="sx-muted">' + esc(t('sess.matrix.lead')) + '</p>';
    if (S.matrixError) out += alertBox(esc(S.matrixError));
    if (m && !m.classes.length) out += '<p style="margin-top:.75rem;">' + esc(t('sess.matrix.empty')) + '</p>';
    if (m && m.classes.length) {
      out += '<div class="sx-table-wrap"><table class="sx-table"><thead><tr><th></th>' +
        m.sessions.map(function (s) { return '<th class="sx-mono">' + esc(s) + '</th>'; }).join('') + '</tr></thead><tbody>';
      m.classes.forEach(function (hz) {
        var row = m.cells[String(hz)] || {};
        out += '<tr><th class="sx-mono">' + hz + ' Hz</th>' + m.sessions.map(function (s) {
          var c = row[s];
          return '<td class="sx-mono">' + (c ? c.accepted + '/' + c.total : '<span class="sx-muted" title="' + esc(t('sess.matrix.empty_cell')) + '">—</span>') + '</td>';
        }).join('') + '</tr>';
      });
      out += '</tbody></table></div>';
      m.classes.forEach(function (hz) {
        var n = m.sessions_per_class[String(hz)] || 0;
        if (n < m.min_sessions_per_class) {
          out += '<p class="sx-warn" style="margin-top:.5rem;">' + esc(plural(n, 'sess.matrix.short_one', 'sess.matrix.short_many', { hz: hz, k: m.min_sessions_per_class })) + '</p>';
        }
      });
    }
    return out + '</section>';
  }

  // ── Events, delegated ─────────────────────────────────────────────────────
  view.addEventListener('click', function (e) {
    var el = e.target.closest('[data-action]');
    if (!el) return;
    var action = el.getAttribute('data-action'), id = el.getAttribute('data-id');
    if (action === 'home') { e.preventDefault(); go(''); }
    else if (action === 'new') go('new');
    else if (action === 'open') go(id);
    else if (action === 'confirm') { S.confirming = id; render(); }
    else if (action === 'cancel') { S.confirming = null; render(); }
    else if (action === 'discard') {
      api.discard(id).then(function () {
        S.drafts = S.drafts.filter(function (d) { return d.draft_id !== id; });
        S.confirming = null; render();
      }, function () { S.listError = t('sess.err.discard'); render(); });
    }
    else if (action === 'reload') loadDraft(false);
    else if (action === 'recalibrate') { S.calibRetry = true; S.calibError = null; render(); }
    else if (action === 'drop-frame') dropFrame(Number(el.getAttribute('data-hz')), el.getAttribute('data-name'));
    else if (action === 'commit') commit(false);
    else if (action === 'commit-replace') commit(true);
    else if (action === 'split') resplit();
  });

  function handleFiles(target, files) {
    if (!files || !files.length) return;
    if (target === 'calib') calibrate(files[0]);
    else if (target.indexOf('freq:') === 0) uploadFrames(Number(target.slice(5)), files);
  }

  view.addEventListener('change', function (e) {
    var target = e.target.getAttribute && e.target.getAttribute('data-files');
    if (target) handleFiles(target, e.target.files);
  });
  view.addEventListener('input', function (e) {
    var name = e.target.getAttribute && e.target.getAttribute('data-field');
    if (name && S.form) S.form[name] = e.target.value;
  });
  view.addEventListener('submit', function (e) {
    if (e.target.id === 'sx-form') { e.preventDefault(); submitForm(); }
  });
  view.addEventListener('dragover', function (e) {
    var zone = e.target.closest('[data-drop]');
    if (zone) { e.preventDefault(); zone.classList.add('is-over'); }
  });
  view.addEventListener('dragleave', function (e) {
    var zone = e.target.closest('[data-drop]');
    if (zone) zone.classList.remove('is-over');
  });
  view.addEventListener('drop', function (e) {
    var zone = e.target.closest('[data-drop]');
    if (!zone) return;
    e.preventDefault();
    zone.classList.remove('is-over');
    handleFiles(zone.getAttribute('data-drop'), e.dataTransfer.files);
  });

  window.addEventListener('hashchange', route);
  document.addEventListener('wittenberg:langchange', function () { render(); renderServer(); });
  document.addEventListener('DOMContentLoaded', function () { checkServer(); route(); });
})();

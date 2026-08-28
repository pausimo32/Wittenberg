/**
 * adapter.js
 * Wittenberg — Traduction du contrat de l'API vers la forme attendue par l'interface.
 *
 * L'API est la source de vérité (docs/05 §3) et c'est le frontend qui s'y adapte. Ce
 * fichier est le seul endroit où sa réponse est traduite : tout le reste de l'interface
 * continue de recevoir la forme qu'il affichait déjà.
 *
 * L'autre solution — un shim /api/v1 côté serveur — aurait créé un second contrat de
 * réponse à tenir à jour en parallèle de celui de docs/05 §3. C'est exactement la copie
 * en double que ce projet interdit, et c'est pourquoi la traduction vit ici.
 *
 * Fonctions pures, sans DOM : elles sont testées par `node --test tests/js/`.
 */
(function (root, factory) {
  var api = factory();
  root.WittenbergAdapter = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // docs/05 §4 — les trois paliers. « La décision d'interface la plus importante du
  // projet » : jamais un nombre seul, jamais une faible confiance masquée.
  var TIER_HIGH = 0.70;
  var TIER_MID = 0.40;

  function tierOf(p) {
    if (p >= TIER_HIGH) return 'high';
    if (p >= TIER_MID) return 'mid';
    return 'low';
  }

  /**
   * Un pourcentage tronqué vers zéro, jamais arrondi vers le haut : 0,6999 doit se lire
   * « 69,9 % » et non « 70 % » sous un titre qui dit « ce pourrait être ». Le lecteur
   * doit toujours voir que le palier et le nombre s'accordent.
   */
  function truncatePercent(p) {
    return Math.floor(p * 1000) / 10;
  }

  /** Une fréquence telle que la liste des classes la stocke. Jamais « environ 1500 ». */
  function formatFrequency(hz) {
    return hz + ' Hz';
  }

  /**
   * L'API ne renvoie pas de libellé : elle renvoie une fréquence entière et un indice de
   * classe. Le libellé est fabriqué ici ; `frequencies.js` n'ajoute que l'éditorial, et
   * seulement s'il connaît cette fréquence. Les classes n'étant pas encore choisies,
   * l'absence de métadonnées locales est le cas normal, pas une erreur.
   */
  function toFrequency(entry) {
    var local = (typeof getFrequencyByHz === 'function') ? getFrequencyByHz(entry.freq_hz) : null;
    var out = {};
    if (local) {
      for (var k in local) {
        if (Object.prototype.hasOwnProperty.call(local, k)) out[k] = local[k];
      }
    }
    out.hz = entry.freq_hz;
    out.label = formatFrequency(entry.freq_hz);
    out.classIndex = entry.class_index;
    out.hasLocalMetadata = Boolean(local);
    return out;
  }

  function mapPrediction(body) {
    var top = (body.top3 || []).slice().sort(function (a, b) { return b.p - a.p; });
    var best = top.length ? top[0] : null;
    var tier = best ? tierOf(best.p) : 'low';
    var quality = body.quality || { quality: null, reason: null, metrics: {} };

    return {
      success: true,
      tier: tier,

      // Au palier bas, le titre ne nomme personne. Afficher « probablement N Hz » sous
      // 0,35 et le rattraper en petits caractères est précisément la faible confiance
      // masquée qu'interdit docs/05 §4.
      leader: (tier === 'low' || !best) ? null : toFrequency(best),
      runnerUp: (tier === 'mid' && top.length > 1) ? toFrequency(top[1]) : null,

      frequency: best ? toFrequency(best) : null,
      confidence: best ? best.p : 0,

      // Les trois candidats sont montrés à tous les paliers, sans exception.
      topPredictions: top.map(function (e) {
        return { frequency: toFrequency(e), confidence: e.p };
      }),

      // La géométrie est un état distinct de la confiance. Une réponse peu sûre et une
      // réponse mal redressée sont deux échecs différents et appellent deux gestes
      // différents : reprendre la photo, ou la reprendre avec le cadre (docs/05 §4).
      geometry: {
        rectified: body.roi_mode === 'homography',
        roiMode: body.roi_mode,
        markersFound: body.markers_found,
      },

      quality: {
        quality: quality.quality,
        reason: quality.reason,
        metrics: quality.metrics || {},
      },

      elapsedMs: body.elapsed_ms,
      timings: body.timings_ms || null,
      stages: body.stages || [],
    };
  }

  /** `/api/health` remplace `/api/v1/model-info`. */
  function mapHealth(health) {
    if (!health) return { available: false, status: 'unreachable', reachable: false, doNotReport: null };
    return {
      available: true,
      status: 'ready',
      reachable: true,
      modelName: health.model,
      modelVersion: health.model_run,
      preprocessingVersion: health.preprocess_version,
      device: health.device,
      classes: health.classes || [],
      synthetic: Boolean(health.synthetic),

      // null quand les chiffres du run sont citables ; une phrase quand ils ne le sont
      // pas. Le bandeau est accroché à ce champ, donc il disparaîtra de lui-même le jour
      // où un vrai checkpoint remplacera le checkpoint synthétique : personne n'aura à
      // penser à l'enlever.
      doNotReport: health.do_not_report || null,
    };
  }

  return {
    TIER_HIGH: TIER_HIGH,
    TIER_MID: TIER_MID,
    tierOf: tierOf,
    truncatePercent: truncatePercent,
    formatFrequency: formatFrequency,
    mapPrediction: mapPrediction,
    mapHealth: mapHealth,
  };
});

/**
 * predictionService.js
 * Wittenberg — Service de prédiction
 *
 * Ce module est le point d'entrée du pipeline d'inférence. Il transmet
 * l'image au backend, qui exécute le prétraitement déterministe puis le
 * modèle CNN, et il traduit la réponse pour l'interface.
 *
 * Aucune simulation n'est produite ici : si le modèle n'est pas disponible
 * sur le serveur, le service renvoie une erreur explicite. Le prétraitement
 * (recadrage de la plaque, niveaux de gris, normalisation) est réalisé côté
 * serveur pour rester strictement identique à celui de l'entraînement — il
 * n'est donc volontairement pas dupliqué dans le navigateur.
 *
 * Contrat du backend : docs/05-webapp-spec.md §3.
 *   GET  /api/health                 ce qui est chargé, et s'il est citable
 *   POST /api/predict?stages=true    multipart/form-data, champ « file »
 *
 * La traduction de ce contrat vers la forme attendue par l'interface vit dans
 * adapter.js, chargé avant ce fichier. Ici il ne reste que le transport, la
 * progression et les erreurs.
 */

// ── États du pipeline ───────────────────────────────────────────────────────
const PredictionState = {
  IDLE:      'idle',
  LOADING:   'loading',
  ANALYZING: 'analyzing',
  DONE:      'done',
  ERROR:     'error'
};

// ── Codes d'erreur ──────────────────────────────────────────────────────────
const PredictionErrorCode = {
  MODEL_UNAVAILABLE: 'model_unavailable',
  NETWORK:           'network_error',
  TIMEOUT:           'timeout',
  EMPTY_FILE:        'empty_file',
  FILE_TOO_LARGE:    'file_too_large',
  UNSUPPORTED:       'unsupported_format',
  UNSUPPORTED_HEIF:  'unsupported_heif',
  CORRUPT:           'corrupt_image'
};

/**
 * Les codes HTTP que renvoie /api/predict, traduits vers les codes que
 * l'interface sait déjà afficher. 415 couvre aussi bien un fichier qui n'est
 * pas une image qu'un HEIC que la machine ne sait pas décoder : le message du
 * serveur distingue les deux, donc on le lit plutôt que de deviner.
 */
const HTTP_ERROR_CODES = {
  400: PredictionErrorCode.EMPTY_FILE,
  413: PredictionErrorCode.FILE_TOO_LARGE,
  415: PredictionErrorCode.UNSUPPORTED,
  422: PredictionErrorCode.CORRUPT
};

function _config() {
  return window.WITTENBERG_CONFIG || {
    API_BASE_URL: '',
    API_PREFIX: '',
    REQUEST_TIMEOUT_MS: 30000,
    apiUrl: function (path) { return this.API_BASE_URL + this.API_PREFIX + path; },
    url: function (path) { return this.API_BASE_URL + path; }
  };
}

function _t(key, fallback) {
  return window.WI18n ? window.WI18n.t(key) : fallback;
}

function _adapter() {
  return window.WittenbergAdapter;
}

async function _readError(response) {
  let code = HTTP_ERROR_CODES[response.status] || ('http_' + response.status);
  let message = null;
  try {
    const body = await response.json();
    const detail = body && body.detail;
    if (typeof detail === 'string') {
      message = detail;
      // `imageio.decode_bytes` nomme le format qu'il a refusé ; un HEIC non
      // décodable a son propre message dans l'interface.
      if (response.status === 415 && /heif|heic/i.test(detail)) {
        code = PredictionErrorCode.UNSUPPORTED_HEIF;
      }
    } else if (body && body.error) {
      code = body.error.code || code;
      message = body.error.message || null;
    }
  } catch (e) {
    // Réponse non JSON (proxy, page d'erreur) : on garde le message générique.
  }
  return { code: code, message: message };
}

// ── Service principal ────────────────────────────────────────────────────────
const PredictionService = {
  /**
   * Interroge le backend sur l'état du modèle.
   *
   * `available: false` couvre aussi bien un backend injoignable qu'un backend
   * démarré sans modèle ; `status` distingue les deux cas. En pratique l'API
   * refuse de démarrer sans checkpoint valide (docs/05 §3), donc « injoignable »
   * est le seul échec que cette page verra.
   */
  async getModelInfo() {
    const config = _config();
    try {
      const response = await fetch(config.url('/api/health'), {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      if (!response.ok) {
        return { available: false, status: 'unreachable', reason: null, reachable: true };
      }
      return _adapter().mapHealth(await response.json());
    } catch (e) {
      return _adapter().mapHealth(null);
    }
  },

  /**
   * Pipeline complet : image → fréquence estimée.
   *
   * @param {File|Blob} imageFile  Fichier image (JPEG, PNG, WebP…)
   * @param {Function}  onProgress Callback(state, message) pour l'interface
   * @returns {Promise<Object>} Résultat normalisé par adapter.js.
   */
  async predict(imageFile, onProgress = () => {}) {
    const config = _config();

    if (!imageFile) {
      return {
        success: false,
        errorCode: PredictionErrorCode.EMPTY_FILE,
        error: _t('pred.no_file', 'Aucune image sélectionnée.')
      };
    }

    onProgress(PredictionState.LOADING, _t('pred.loading', 'Envoi de l\'image…'));

    const formData = new FormData();
    // Le champ s'appelle « file » : c'est le nom qu'attend /api/predict. Le nom de
    // fichier est neutralisé côté serveur ; il ne sert qu'au journal du navigateur.
    formData.append('file', imageFile, imageFile.name || 'capture.jpg');

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      config.REQUEST_TIMEOUT_MS || 30000
    );

    let response;
    try {
      onProgress(PredictionState.ANALYZING, _t('pred.analysing', 'Analyse du motif en cours…'));
      // stages=true : l'analyseur affiche le visualiseur de pipeline (docs/05 §8) sous
      // le résultat, donc il lui faut les treize étapes. Elles coûtent ≈ 43 ms d'encodage
      // et ≈ 1,8 Mo, ce qui n'est rien sur 127.0.0.1 — et c'est la même requête qui
      // produit la réponse : §3 interdit de faire tourner le pipeline une seconde fois
      // pour l'affichage. `?stages=false` reste servi par l'API pour tout autre appelant.
      response = await fetch(config.url('/api/predict?stages=true'), {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
    } catch (err) {
      clearTimeout(timeoutId);
      onProgress(PredictionState.ERROR, _t('pred.error', 'Erreur lors de l\'analyse'));
      const aborted = err && err.name === 'AbortError';
      return {
        success: false,
        errorCode: aborted ? PredictionErrorCode.TIMEOUT : PredictionErrorCode.NETWORK,
        error: aborted
          ? _t('pred.timeout', 'L\'analyse a pris trop de temps. Réessayez.')
          : _t('pred.unreachable', 'Le service d\'analyse est injoignable. Vérifiez que le backend est démarré.')
      };
    }
    clearTimeout(timeoutId);

    if (!response.ok) {
      const failure = await _readError(response);
      onProgress(PredictionState.ERROR, _t('pred.error', 'Erreur lors de l\'analyse'));
      return {
        success: false,
        status: response.status,
        errorCode: failure.code,
        error: failure.message || _t('pred.unknown', 'Erreur inconnue lors de l\'analyse.')
      };
    }

    const body = await response.json();
    onProgress(PredictionState.DONE, _t('pred.done', 'Analyse terminée'));

    // Toute la traduction du contrat vit dans adapter.js — un seul endroit.
    return _adapter().mapPrediction(body);
  }
};

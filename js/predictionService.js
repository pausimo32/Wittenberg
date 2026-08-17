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
 * Contrat du backend : voir backend/README.md
 *   GET  /health
 *   GET  /ready
 *   GET  /api/v1/model-info
 *   POST /api/v1/predict   (multipart/form-data, champ « image »)
 */

// ── États du pipeline ───────────────────────────────────────────────────────
const PredictionState = {
  IDLE:      'idle',
  LOADING:   'loading',
  ANALYZING: 'analyzing',
  DONE:      'done',
  ERROR:     'error'
};

// ── Codes d'erreur renvoyés par le backend ──────────────────────────────────
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

function _config() {
  return window.WITTENBERG_CONFIG || {
    API_BASE_URL: 'http://localhost:8000',
    API_PREFIX: '/api/v1',
    REQUEST_TIMEOUT_MS: 30000,
    apiUrl: function (path) { return this.API_BASE_URL + this.API_PREFIX + path; },
    url: function (path) { return this.API_BASE_URL + path; }
  };
}

function _t(key, fallback) {
  return window.WI18n ? window.WI18n.t(key) : fallback;
}

/**
 * Complète une classe renvoyée par le backend avec les métadonnées locales
 * (mode vibratoire, description) lorsqu'une correspondance existe.
 *
 * Le backend reste la seule source de vérité pour la fréquence et le libellé ;
 * `frequencies.js` ne fournit qu'un enrichissement éditorial facultatif. Si le
 * manifeste du modèle décrit des fréquences absentes du fichier local, le
 * résultat s'affiche sans description plutôt qu'avec une description erronée.
 */
function _toFrequency(entry) {
  const hz = entry.frequency_hz;
  const local = (typeof getFrequencyByHz === 'function') ? getFrequencyByHz(hz) : null;
  return Object.assign({}, local || {}, {
    hz: hz,
    label: entry.label,
    classIndex: entry.class_index,
    hasLocalMetadata: Boolean(local)
  });
}

async function _readError(response) {
  let code = 'http_' + response.status;
  let message = null;
  try {
    const body = await response.json();
    if (body && body.error) {
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
   * @returns {Promise<Object>} { available, status, reason, classes, ... }
   *   `available: false` couvre aussi bien un backend injoignable qu'un
   *   backend démarré sans modèle : `status` distingue les deux cas.
   */
  async getModelInfo() {
    const config = _config();
    try {
      const response = await fetch(config.apiUrl('/model-info'), {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      if (!response.ok) {
        return { available: false, status: 'unreachable', reason: null, reachable: true };
      }
      const info = await response.json();
      info.reachable = true;
      return info;
    } catch (e) {
      return { available: false, status: 'unreachable', reason: null, reachable: false };
    }
  },

  /**
   * Pipeline complet : image → fréquence estimée.
   *
   * @param {File|Blob} imageFile  Fichier image (JPEG, PNG, WebP…)
   * @param {Function}  onProgress Callback(state, message) pour l'interface
   * @returns {Promise<Object>} Résultat normalisé pour l'interface.
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
    // Le nom transmis est neutralisé côté serveur ; il ne sert qu'au journal
    // de bord du navigateur.
    formData.append('image', imageFile, imageFile.name || 'capture.jpg');

    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      config.REQUEST_TIMEOUT_MS || 30000
    );

    let response;
    try {
      onProgress(PredictionState.ANALYZING, _t('pred.analysing', 'Analyse du motif en cours…'));
      response = await fetch(config.apiUrl('/predict'), {
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

    const quality = body.quality || { score: null, warnings: [] };

    return {
      success: true,
      requestId: body.request_id,
      modelName: body.model ? body.model.name : null,
      modelVersion: body.model ? body.model.version : null,
      preprocessingVersion: body.preprocessing ? body.preprocessing.version : null,
      processingMs: body.processing_ms,

      frequency: _toFrequency(body.prediction),
      confidence: body.prediction.confidence,
      topPredictions: (body.top_k || []).map(function (entry) {
        return { frequency: _toFrequency(entry), confidence: entry.probability };
      }),

      lowConfidence: body.prediction.reliable === false,
      reason: body.prediction.reason || null,
      margin: body.prediction.margin,

      quality: {
        score: quality.score,
        warnings: quality.warnings || []
      }
    };
  }
};

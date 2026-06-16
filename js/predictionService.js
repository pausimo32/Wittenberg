/**
 * predictionService.js
 * Wittenberg — Service de prédiction (simulation)
 *
 * Ce module est le point d'entrée du pipeline d'inférence.
 * Actuellement, il implémente une simulation réaliste.
 *
 * POUR CONNECTER UN VRAI MODÈLE :
 * Remplacez la fonction `_mockInference()` par l'une des options :
 *   1. TensorFlow.js local : tf.loadGraphModel(MODEL_URL) + model.predict()
 *   2. ONNX Runtime Web  : ort.InferenceSession.create(MODEL_PATH)
 *   3. API REST          : fetch('/api/predict', { body: formData })
 */

// ── Configuration ──────────────────────────────────────────────────────────
const PREDICTION_CONFIG = {
  // Résolution cible pour le prétraitement de l'image
  INPUT_SIZE: 224,
  // Seuil de confiance minimum pour valider une prédiction
  CONFIDENCE_THRESHOLD: 0.55,
  // Latence simulée de l'inférence (ms) — à supprimer avec un vrai modèle
  MOCK_LATENCY_MS: 2200,
  // Version du modèle simulé
  MODEL_VERSION: 'mock-v0.1',
  // Le vrai modèle n'est pas encore connecté
  MODEL_CONNECTED: false
};

// ── État ────────────────────────────────────────────────────────────────────
const PredictionState = {
  IDLE:      'idle',
  LOADING:   'loading',
  ANALYZING: 'analyzing',
  DONE:      'done',
  ERROR:     'error'
};

// ── Utilitaires de prétraitement ─────────────────────────────────────────────
const ImagePreprocessor = {
  /**
   * Charge une image depuis un fichier Blob/File et retourne un objet canvas.
   * Étape 1 du pipeline : redimensionnement + normalisation.
   */
  async fromBlob(blob) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(blob);
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width  = PREDICTION_CONFIG.INPUT_SIZE;
        canvas.height = PREDICTION_CONFIG.INPUT_SIZE;
        const ctx = canvas.getContext('2d');
        // Remplir en noir pour les bords si l'image n'est pas carrée
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // Dessiner centré et mis à l'échelle
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const dx = (canvas.width  - img.width  * scale) / 2;
        const dy = (canvas.height - img.height * scale) / 2;
        ctx.drawImage(img, dx, dy, img.width * scale, img.height * scale);
        URL.revokeObjectURL(url);
        resolve(canvas);
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error(window.WI18n ? window.WI18n.t('pred.img_error') : 'Impossible de charger l\'image.')); };
      img.src = url;
    });
  },

  /**
   * Convertit le canvas en niveaux de gris et normalise les pixels [0,1].
   * Étape 2 du pipeline : conversion niveaux de gris.
   * @returns {Float32Array} tenseur (H × W × 1)
   */
  toGrayscaleTensor(canvas) {
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    const tensor = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      // Luminance perceptuelle
      tensor[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    }
    return tensor;
  },

  /**
   * Normalisation de contraste (étirement d'histogramme simplifié).
   * Étape 3 du pipeline : amélioration du contraste.
   */
  normalizeContrast(tensor) {
    let min = Infinity, max = -Infinity;
    for (const v of tensor) { if (v < min) min = v; if (v > max) max = v; }
    const range = max - min || 1;
    return tensor.map(v => (v - min) / range);
  }
};

// ── Inférence simulée ────────────────────────────────────────────────────────
/**
 * Simule la sortie d'un CNN de classification.
 * Retourne un vecteur de probabilités pour chaque classe de fréquence.
 * À REMPLACER par le modèle réel.
 */
function _mockInference(tensor) {
  // Utilise la moyenne des pixels comme "signal" pour varier les résultats
  const mean = tensor.reduce((s, v) => s + v, 0) / tensor.length;
  const seed = Math.round(mean * 1e6);
  const rng = _seededRandom(seed);

  const numClasses = FREQUENCIES.length;
  const logits = Array.from({ length: numClasses }, () => rng() * 4 - 2);

  // Ajouter un pic aléatoire pour simuler une prédiction dominante
  const peakIdx = Math.floor(rng() * numClasses);
  logits[peakIdx] += rng() * 4 + 2;

  // Softmax
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((s, v) => s + v, 0);
  return exps.map(e => e / sumExp);
}

function _seededRandom(seed) {
  let s = seed;
  return function() {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ── Service principal ────────────────────────────────────────────────────────
const PredictionService = {
  /**
   * Pipeline complet : image → fréquence estimée.
   *
   * @param {File|Blob} imageFile    Fichier image (JPEG/PNG recommandé)
   * @param {Function}  onProgress   Callback(state, message) pour les mises à jour UI
   * @returns {Promise<PredictionResult>}
   */
  async predict(imageFile, onProgress = () => {}) {
    onProgress(PredictionState.LOADING, window.WI18n ? window.WI18n.t('pred.loading') : 'Chargement de l\'image…');

    let canvas, tensor;

    try {
      // ── Étape 1 : Redimensionnement ──────────────────────────────────────
      canvas = await ImagePreprocessor.fromBlob(imageFile);

      onProgress(PredictionState.ANALYZING, window.WI18n ? window.WI18n.t('pred.preprocess') : 'Prétraitement en cours…');

      // ── Étape 2 : Conversion niveaux de gris ─────────────────────────────
      tensor = ImagePreprocessor.toGrayscaleTensor(canvas);

      // ── Étape 3 : Normalisation du contraste ─────────────────────────────
      tensor = ImagePreprocessor.normalizeContrast(tensor);

      onProgress(PredictionState.ANALYZING, window.WI18n ? window.WI18n.t('pred.analysing') : 'Analyse du motif en cours…');

      // ── Étape 4 : Inférence ───────────────────────────────────────────────
      // Simuler une latence de traitement réseau/GPU
      await _delay(PREDICTION_CONFIG.MOCK_LATENCY_MS);

      // === ICI : Remplacer par l'appel au vrai modèle ===
      // Exemple TensorFlow.js :
      //   const inputTensor = tf.tensor4d(tensor, [1, 224, 224, 1]);
      //   const outputTensor = model.predict(inputTensor);
      //   const probabilities = await outputTensor.data();
      // =====================================================
      const probabilities = _mockInference(tensor);

      // ── Étape 5 : Décodage des résultats ─────────────────────────────────
      const topK = _getTopK(probabilities, 3);
      const best = topK[0];

      onProgress(PredictionState.DONE, window.WI18n ? window.WI18n.t('pred.done') : 'Analyse terminée');

      return {
        success: true,
        isMock: !PREDICTION_CONFIG.MODEL_CONNECTED,
        modelVersion: PREDICTION_CONFIG.MODEL_VERSION,
        frequency: FREQUENCIES[best.index],
        confidence: best.probability,
        topPredictions: topK.map(k => ({
          frequency: FREQUENCIES[k.index],
          confidence: k.probability
        })),
        lowConfidence: best.probability < PREDICTION_CONFIG.CONFIDENCE_THRESHOLD,
        warning: !PREDICTION_CONFIG.MODEL_CONNECTED
          ? (window.WI18n ? window.WI18n.t('analyser.mock_warn') : 'Aucun modèle entraîné n\'est encore connecté à cette version. Cette estimation est une simulation destinée à valider l\'interface.')
          : null
      };

    } catch (err) {
      onProgress(PredictionState.ERROR, window.WI18n ? window.WI18n.t('pred.error') : 'Erreur lors de l\'analyse');
      return {
        success: false,
        error: err.message || (window.WI18n ? window.WI18n.t('pred.unknown') : 'Erreur inconnue lors de l\'analyse.')
      };
    }
  }
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function _delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function _getTopK(probabilities, k) {
  return probabilities
    .map((p, i) => ({ index: i, probability: p }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
}

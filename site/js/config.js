/**
 * config.js
 * Wittenberg — Configuration de l'application côté navigateur
 *
 * Le frontend est un site statique : il n'existe ni build ni variables
 * d'environnement. L'adresse de l'API se configure donc, par ordre de priorité :
 *
 *   1. une balise dans la page :
 *        <meta name="wittenberg-api-base" content="https://api.exemple.org">
 *   2. une valeur définie avant le chargement de ce script :
 *        <script>window.WITTENBERG_API_BASE_URL = 'https://api.exemple.org';</script>
 *   3. une valeur enregistrée dans le navigateur (utile pour les tests) :
 *        localStorage.setItem('wittenberg.apiBaseUrl', 'http://localhost:8000')
 *   4. la valeur par défaut ci-dessous.
 *
 * En développement (localhost, 127.0.0.1 ou ouverture directe du fichier),
 * l'API est supposée écouter sur http://localhost:8000.
 * En production, une chaîne vide signifie « même origine » : le backend est
 * alors servi derrière le même domaine, ce qui évite toute question de CORS.
 */
(function () {
  'use strict';

  // Même origine dans les deux cas : FastAPI sert ce site (docs/05 §3). Il n'y a plus
  // de « développement » où le backend vivrait sur un autre port — `python tasks.py app`
  // sert les pages et l'API depuis 127.0.0.1:8000. Cela supprime la question CORS au
  // lieu d'y répondre par un joker, que src/api.py refuse par conception.
  var DEV_API_BASE  = '';
  var PROD_API_BASE = '';
  var STORAGE_KEY   = 'wittenberg.apiBaseUrl';

  function fromMeta() {
    var meta = document.querySelector('meta[name="wittenberg-api-base"]');
    return meta ? meta.getAttribute('content') : null;
  }

  function fromStorage() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      // Mode privé ou stockage bloqué : sans conséquence.
      return null;
    }
  }

  function isLocalEnvironment() {
    var host = window.location.hostname;
    return (
      window.location.protocol === 'file:' ||
      host === 'localhost' ||
      host === '127.0.0.1' ||
      host === '::1' ||
      host === ''
    );
  }

  function resolveBaseUrl() {
    var candidate =
      fromMeta() ||
      window.WITTENBERG_API_BASE_URL ||
      fromStorage() ||
      (isLocalEnvironment() ? DEV_API_BASE : PROD_API_BASE);

    // Supprimer la barre oblique finale pour éviter les doubles séparateurs.
    return (candidate || '').replace(/\/+$/, '');
  }

  window.WITTENBERG_CONFIG = {
    /** Racine de l'API, sans barre oblique finale. '' signifie même origine. */
    API_BASE_URL: resolveBaseUrl(),

    /** L'API n'est pas versionnée : docs/05 §3 définit /api/health et /api/predict. */
    API_PREFIX: '',

    /** Délai maximal d'une requête d'analyse, en millisecondes. */
    REQUEST_TIMEOUT_MS: 30000,

    /** Nombre d'hypothèses affichées dans le résultat. */
    TOP_K: 3,

    /** Construit une URL absolue vers une route de l'API. */
    url: function (path) {
      return this.API_BASE_URL + path;
    },

    /** Construit une URL absolue vers une route versionnée. */
    apiUrl: function (path) {
      return this.API_BASE_URL + this.API_PREFIX + path;
    }
  };
})();

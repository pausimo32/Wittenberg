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

  var DEV_API_BASE  = 'http://localhost:8000';
  var PROD_API_BASE = '';        // même origine
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

    /** Préfixe versionné des routes d'inférence. */
    API_PREFIX: '/api/v1',

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

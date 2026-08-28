/**
 * banner.js
 * Wittenberg — Le bandeau « checkpoint synthétique », sur toutes les pages.
 *
 * `/api/health` porte `do_not_report` : null quand les chiffres du run sont citables,
 * une phrase quand ils ne le sont pas (docs/05 §3). `src/train.py` écrit ce champ dans
 * les métriques de tout run construit sur des sessions SYNTH-*, où la précision mesure
 * à quel point le réseau a inversé la formule qui a dessiné les images — et rien du
 * tout au sujet d'une plaque.
 *
 * L'application sert volontiers un tel checkpoint ; ce qu'elle ne doit pas faire, c'est
 * laisser le nombre à l'écran ressembler à un résultat. D'où ce bandeau, sur toutes les
 * pages et non sur le seul analyseur : les tableaux de données et les pages de méthode
 * décrivent le même modèle.
 *
 * Il est accroché au champ, jamais à une constante : le jour où un vrai checkpoint
 * remplacera celui-ci, `do_not_report` deviendra null et le bandeau disparaîtra de
 * lui-même. Personne n'aura à penser à l'enlever.
 */
(function () {
  'use strict';

  var STYLE = [
    'background:#92400e',
    'color:#fff',
    'padding:.625rem 1rem',
    'font-size:.8125rem',
    'line-height:1.5',
    'text-align:center',
    'position:relative',
    'z-index:60'
  ].join(';');

  function render(text) {
    var bar = document.createElement('div');
    bar.id = 'synthetic-banner';
    bar.setAttribute('role', 'status');
    bar.style.cssText = STYLE;

    var label = (window.WI18n && window.WI18n.t)
      ? window.WI18n.t('analyser.synthetic_banner')
      : 'Modèle entraîné sur des images synthétiques : aucun chiffre affiché ici ne décrit une plaque réelle.';

    var strong = document.createElement('strong');
    strong.textContent = label;
    bar.appendChild(strong);

    // La phrase du serveur est ajoutée telle quelle : elle dit précisément ce que la
    // précision de ce run mesure, et c'est le run lui-même qui la porte.
    var detail = document.createElement('div');
    detail.style.cssText = 'font-weight:400;opacity:.9;margin-top:.25rem;';
    detail.textContent = text;
    bar.appendChild(detail);

    document.body.insertBefore(bar, document.body.firstChild);
  }

  function check() {
    var config = window.WITTENBERG_CONFIG;
    var url = (config ? config.url('/api/health') : '/api/health');

    fetch(url, { headers: { 'Accept': 'application/json' } })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (health) {
        var info = window.WittenbergAdapter.mapHealth(health);
        if (info.doNotReport) render(info.doNotReport);
      })
      .catch(function () {
        // Backend injoignable : pas de bandeau. Une absence de bandeau sur une API
        // morte n'est pas un mensonge — aucun chiffre n'est affiché non plus — alors
        // qu'une boîte d'erreur sur chacune des cinq pages le serait.
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', check);
  } else {
    check();
  }
})();

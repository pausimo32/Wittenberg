/**
 * pipelineViewer.js
 * Wittenberg — Le visualiseur de pipeline (docs/05-webapp-spec.md §8)
 *
 * Montre à l'utilisateur les treize étapes que la machine a suivies pour transformer sa
 * photo en une fréquence : les huit du prétraitement, les trois blocs du réseau, la carte
 * Grad-CAM et la distribution de sortie.
 *
 * **Une seule requête, un seul passage du pipeline.** Les treize étapes arrivent dans la
 * réponse qui a produit le résultat affiché à côté. Rien n'est recalculé pour l'affichage
 * (§3, §8) : les images montrées sont littéralement celles qui ont produit la réponse.
 *
 * **Animé, jamais diffusé en flux.** Mesuré, le pipeline entier prend ≈ 1,3 ms ; les treize
 * étapes arriveraient dans une même trame d'animation et l'utilisateur ne verrait que la
 * dernière. §8 tranche : une seule requête, et l'interface déroule les étapes à un rythme
 * humain — ≈ 1,1 s par étape, avec pause, flèches et rail de miniatures.
 *
 * **Le temps affiché est le temps mesuré.** Chaque étape porte le coût réel de ce
 * traitement-ci (`ms`), mesuré là où le travail a eu lieu. Une étape que personne n'a
 * chronométrée affiche un tiret et jamais un nombre vraisemblable.
 *
 * Ce fichier se divise en deux moitiés. La première est faite de fonctions pures, sans
 * DOM, et porte toutes les règles de §8 ; elle est testée par `node --test
 * "tests/js/*.test.js"`. La seconde ne fait qu'obéir : elle peint, et ne décide rien.
 */
(function (root, factory) {
  // `truncatePercent` vit dans adapter.js et n'est pas recopiée ici : la troncature vers
  // zéro est la règle d'honnêteté de §4, et une seconde copie dériverait en silence.
  var adapter = (typeof module !== 'undefined' && module.exports)
    ? require('./adapter.js')
    : root.WittenbergAdapter;
  var api = factory(adapter);
  root.WittenbergPipelineViewer = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (adapter) {
  'use strict';

  // ── Les règles de §8 ──────────────────────────────────────────────────────

  /** §8 : « la UI déroule les étapes à un rythme humain — ~1,1 s par étape ». */
  var DWELL_MS = 1100;

  /**
   * Les deux étapes sur lesquelles l'animation s'attarde.
   *
   * Canny parce que §8 le dit : « l'étape 6 est le déclic, et c'est celle sur laquelle
   * l'animation doit s'attarder » — c'est là que tout disparaît sauf les lignes nodales.
   * Grad-CAM parce que §8 l'appelle « l'image la plus didactique du projet » : c'est elle
   * qui transforme « la machine dit 860 Hz » en « la machine dit 860 Hz à cause de ces
   * lignes-là ».
   */
  var LINGER_MS = { edges: 2400, gradcam: 2000 };

  /**
   * §8 : les étapes 7 et 8 se rendent sans lissage. À 128×128 la grille de pixels *est*
   * l'information — l'interpoler cacherait que le réseau voit quelque chose de grossier.
   */
  var PIXELATED = { resized: true, tensor: true };

  /** Les étapes voyagent en PNG base64 dans la réponse : §3 n'écrit aucune image sur disque. */
  var IMAGE_PREFIX = 'data:image/png;base64,';

  /** Le libellé vient de la clé, jamais de `stage.title`. */
  function labelKey(stage) {
    // L'API répond en catalan (`Escala de grisos`) ; l'interface est française par défaut
    // avec un sélecteur ca/es/fr (ADR 0007). Lire `stage.title` figerait une langue sur
    // cet écran et ferait mentir le sélecteur.
    return 'viewer.stage.' + stage.key;
  }

  function captionKey(stage) {
    return 'viewer.caption.' + stage.key;
  }

  /**
   * Le temps mesuré de cette étape, sur cette requête.
   *
   * `null` devient un tiret : §8 refuse qu'un chiffre que personne n'a mesuré apparaisse à
   * côté d'un chiffre mesuré, parce que le lecteur ne peut pas distinguer les deux. Et une
   * étape trop rapide pour deux décimales se dit petite plutôt que nulle — `0,00 ms` se
   * lirait « gratuit », ce qui est une affirmation différente de « moins d'un centième ».
   */
  function formatMs(ms) {
    if (ms === null || ms === undefined) return '—';
    if (ms === 0) return '0 ms';
    if (ms < 0.01) return '< 0,01 ms';
    return ms.toFixed(ms < 10 ? 2 : 1).replace('.', ',') + ' ms';
  }

  /** `[512, 512]` → `512 × 512`. */
  function formatShape(shape) {
    return (shape || []).join(' × ');
  }

  /**
   * Les deux tailles de l'étape 1, ou `null` pour toute étape montrée à sa vraie taille.
   *
   * §8 plafonne l'étape 1 à 768 px pour l'affichage — non par cosmétique : encoder une
   * capture 4032×3024 non plafonnée coûte 459 ms et 24 Mo, soit trois fois le budget de §7
   * dès la première vraie photo de téléphone. Le pipeline, lui, reçoit toujours le tableau
   * pleine résolution. L'interface doit donc dire les deux tailles ; montrer une image de
   * 768 px sous une étiquette « 4032 px » serait un faux compte rendu de la capture.
   */
  function shapeNote(stage) {
    if (!stage.display_shape) return null;
    // Seul un vrai rétrécissement se dit. L'étape 8 est un `(128, 128, 1)` float32 encodé
    // en image `(128, 128)` : les deux formes diffèrent sans qu'un pixel soit perdu, et
    // l'annoncer comme « montrée en … · originale … » proclamerait un plafonnement qui
    // n'a pas eu lieu. La note appartient à la capture, qui elle est vraiment réduite.
    if (stage.display_shape[0] >= stage.shape[0] && stage.display_shape[1] >= stage.shape[1]) {
      return null;
    }
    return {
      displayed: formatShape(stage.display_shape),
      original: formatShape(stage.shape),
    };
  }

  /**
   * Les paramètres de l'étape, tels que `config/preprocess.json` les contient.
   *
   * §8 : « Montrer l'appel OpenCV exact et les paramètres *lus depuis
   * config/preprocess.json*, non retapés dans l'interface. Si quelqu'un change un seuil, le
   * visualiseur doit annoncer le nouveau nombre. » Les nombres sont donc imprimés tels
   * quels, sans arrondi : `scale` vaut 0.00392156862745098 et l'afficher « 0,0039 » romprait
   * cette promesse précisément là où elle compte.
   */
  function formatParams(params) {
    if (!params) return [];
    var out = [];
    Object.keys(params).forEach(function (name) {
      out.push({ name: name, value: formatValue(params[name]) });
    });
    return out;
  }

  /**
   * Les mêmes paramètres, sur une ligne — la forme qu'attend le panneau superposé.
   *
   * §8 exige les paramètres du config à l'écran quelle que soit la mise en page. Le
   * panneau n'a pas la place d'un tableau, donc il les met sur une ligne ; il ne les
   * supprime pas pour rester net.
   */
  function formatParamsLine(params) {
    return formatParams(params)
      .map(function (entry) { return entry.name + '=' + entry.value; })
      .join(' · ');
  }

  function formatValue(value) {
    // Une paire de nombres est une géométrie — un noyau, une grille, une taille — et se lit
    // comme telle. Les booléens et les constantes OpenCV restent des littéraux de code :
    // `INTER_AREA` et `false` ne se traduisent pas.
    if (Array.isArray(value)) {
      return value.length === 2 ? value.join(' × ') : value.join(', ');
    }
    if (value !== null && typeof value === 'object') return JSON.stringify(value);
    return String(value);
  }

  /**
   * « 512 × 512 · uint8 », ou un tiret pour une étape qui n'est pas un tableau.
   *
   * L'étape 13 est un diagramme en barres : elle ne porte ni `shape` ni `dtype`, et
   * assembler deux champs absents affichait un « · » seul dans la fiche.
   */
  function formatDimensions(stage) {
    var parts = [];
    if (stage.shape && stage.shape.length) parts.push(formatShape(stage.shape));
    if (stage.dtype) parts.push(stage.dtype);
    return parts.length ? parts.join(' · ') : '—';
  }

  function isPixelated(stage) {
    return PIXELATED[stage.key] === true;
  }

  /**
   * L'image unique que le panneau de résultat montre à côté de la fréquence : l'étape 7,
   * ce que le réseau reçoit réellement.
   *
   * Elle est trouvée **par sa clé**. La prendre comme « la dernière du tableau » ne tenait
   * que tant que l'analyseur demandait `?stages=false` et recevait une liste d'un seul
   * élément ; demander les treize étapes fait de la dernière la distribution de sortie,
   * qui ne porte aucune image.
   */
  function edgeMapSrc(stages) {
    var found = null;
    (stages || []).forEach(function (stage) {
      if (stage.key === 'resized' && stage.png_base64) found = stage;
    });
    return found ? IMAGE_PREFIX + found.png_base64 : null;
  }

  function dwellMs(stage) {
    return LINGER_MS[stage.key] || DWELL_MS;
  }

  /** « 1 » → « 01 » : la maquette aligne les rangs sur deux chiffres. */
  function formatRank(n) {
    var s = String(n);
    return s.length < 2 ? '0' + s : s;
  }

  /**
   * Le rang de l'étape courante en pourcentage — la largeur de la barre du pied.
   *
   * Elle compte les étapes franchies, pas les intervalles : la première étape remplit
   * déjà un treizième plutôt que rien du tout, et la dernière remplit la barre. Une
   * barre encore vide pendant la première seconde du film se lirait « rien n'a
   * commencé », ce qui est faux — l'étape 1 est à l'écran.
   */
  function progressPercent(index, total) {
    if (!total || total < 1) return 0;
    var i = Math.min(Math.max(0, Math.floor(index) || 0), total - 1);
    return ((i + 1) / total) * 100;
  }

  /**
   * Le film est-il fini : la dernière étape, et plus rien qui avance.
   *
   * C'est ce qui décide entre « Suivant » et « Voir le résultat ». Être sur la dernière
   * étape ne suffit pas — la lecture peut y arriver toute seule, mais l'utilisateur
   * peut aussi être en pause à mi-course. `playback.next` pose `playing` à faux en
   * arrivant au bout, donc les deux conditions se lisent ensemble.
   */
  function isFinished(index, total, playing) {
    return total > 0 && index >= total - 1 && !playing;
  }

  /**
   * L'automate de lecture : quelle étape est à l'écran, et est-ce que ça avance tout seul.
   *
   * Il ne connaît ni le DOM ni les étapes elles-mêmes — seulement leur nombre. Le
   * minuteur vit dans la moitié DOM, qui lui demande combien de temps tenir l'étape
   * courante ; l'automate, lui, ne décide que du déplacement.
   */
  function playback(count) {
    var total = Math.max(0, Math.floor(count) || 0);

    var self = {
      index: 0,
      playing: false,

      play: function () {
        // Une seule étape (ou aucune) n'est pas un film : `?stages=false` ne renvoie que
        // la carte de contours finale, et une analyse en échec ne renvoie rien.
        if (total > 1) self.playing = true;
        return self;
      },

      pause: function () {
        self.playing = false;
        return self;
      },

      toggle: function () {
        return self.playing ? self.pause() : self.play();
      },

      /**
       * L'étape suivante. `{ manual: true }` quand c'est l'utilisateur qui a appuyé : il
       * vient de prendre la main, donc la lecture automatique s'arrête.
       */
      next: function (options) {
        if (self.index >= total - 1) {
          // Le film s'arrête sur la dernière étape au lieu de reboucler : seize secondes
          // en boucle sur le portable partagé d'une session de terrain sont du bruit.
          self.playing = false;
          return self;
        }
        self.index += 1;
        if (options && options.manual) self.playing = false;
        return self;
      },

      prev: function () {
        if (self.index > 0) self.index -= 1;
        self.playing = false;
        return self;
      },

      /** Le rail. Cliquer une miniature, c'est prendre la main — donc mettre en pause. */
      go: function (i) {
        self.index = Math.min(Math.max(0, Math.floor(i) || 0), Math.max(0, total - 1));
        self.playing = false;
        return self;
      },

      restart: function () {
        self.index = 0;
        return self.play();
      },
    };

    return self;
  }

  // ── La moitié DOM : elle peint, elle ne décide rien ────────────────────────

  function _t(key, fallback) {
    var i18n = (typeof window !== 'undefined') ? window.WI18n : null;
    return (i18n && i18n.t(key)) || fallback || key;
  }

  /** `prefers-reduced-motion` gouverne deux choses : le film, et le défilement de la file. */
  function prefersStill() {
    return typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false;
  }

  /**
   * Accroche le visualiseur au `<dialog>` d'`analyseur.html`.
   *
   * Renvoie `{ show, hide }`. `show(stages)` ouvre le panneau et lance le film ;
   * `hide()` le referme et arrête tout minuteur en cours.
   *
   * Le panneau est un `<dialog>` ouvert par `showModal()` : le piège à focus, la touche
   * Échap et le fond assombri viennent du navigateur. Rien de tout cela n'est réécrit
   * ici, et la fermeture par Échap passe par l'événement `close` comme les autres.
   */
  function mount(root) {
    if (!root) return { show: function () {}, hide: function () {} };

    var el = {
      figure:     root.querySelector('#pv-figure'),
      image:      root.querySelector('#pv-image'),
      bars:       root.querySelector('#pv-bars'),
      compare:    root.querySelector('#pv-compare'),
      compareImg: root.querySelector('#pv-compare-img'),
      compareBtn: root.querySelector('#pv-compare-toggle'),
      counter:    root.querySelector('#pv-counter'),
      note:       root.querySelector('#pv-note'),
      title:      root.querySelector('#pv-title'),
      caption:    root.querySelector('#pv-caption'),
      call:       root.querySelector('#pv-call'),
      params:     root.querySelector('#pv-params'),
      ms:         root.querySelector('#pv-ms'),
      rail:       root.querySelector('#pv-rail'),
      progress:   root.querySelector('#pv-progress'),
      progressTx: root.querySelector('#pv-progress-text'),
      play:       root.querySelector('#pv-play'),
      prev:       root.querySelector('#pv-prev'),
      next:       root.querySelector('#pv-next'),
      done:       root.querySelector('#pv-done'),
      close:      root.querySelector('#pv-close'),
      live:       root.querySelector('#pv-live'),
    };

    var stages = [];
    var deck = playback(0);
    var timer = null;

    function clearTimer() {
      if (timer !== null) { clearTimeout(timer); timer = null; }
    }

    function schedule() {
      clearTimer();
      if (!deck.playing) return;
      timer = setTimeout(function () {
        deck.next();
        render();
      }, dwellMs(stages[deck.index]));
    }

    function render() {
      var stage = stages[deck.index];
      if (!stage) return;

      if (stage.kind === 'bars') {
        el.figure.hidden = true;
        el.bars.hidden = false;
        renderBars(stage);
      } else {
        el.bars.hidden = true;
        el.figure.hidden = false;
        el.image.src = IMAGE_PREFIX + stage.png_base64;
        el.image.alt = _t(captionKey(stage), '');
        el.image.style.imageRendering = isPixelated(stage) ? 'pixelated' : 'auto';
      }

      el.counter.textContent = formatRank(stage.n) + ' / ' + formatRank(stages.length);
      el.title.textContent = _t(labelKey(stage), stage.key);
      el.caption.textContent = _t(captionKey(stage), '');
      el.call.textContent = stage.call;
      el.ms.textContent = formatMs(stage.ms);
      // Le tiret du temps est voulu (§8 : une étape non chronométrée le dit). Celui des
      // dimensions ne veut rien dire une fois son étiquette partie : l'étape 13 est un
      // diagramme en barres, elle n'a pas de forme, et la note s'efface simplement.
      var dimensions = shapeText(stage);
      el.note.hidden = (dimensions === '—');
      el.note.textContent = dimensions;

      // Les paramètres du config, sur une ligne. §8 les veut à l'écran quelle que soit
      // la place disponible ; ils sont resserrés, jamais supprimés.
      var line = formatParamsLine(stage.params);
      el.params.textContent = line ? ' · ' + line : '';

      renderRail();

      // L'avancement : la barre, puis la même chose en toutes lettres pour qui ne lit
      // pas une barre — un rang seul (« 6 / 13 ») ne dit pas ce qu'on regarde.
      el.progress.style.width = progressPercent(deck.index, stages.length) + '%';
      el.progressTx.textContent = _t('viewer.progress', 'Phase {n} sur {total} — {t}')
        .replace('{n}', String(stage.n))
        .replace('{total}', String(stages.length))
        .replace('{t}', _t(labelKey(stage), stage.key));

      // Au bout du film « Suivant » n'a plus de sens : la maquette le remplace par la
      // sortie. Les deux ne sont jamais visibles ensemble.
      var done = isFinished(deck.index, stages.length, deck.playing);
      el.next.hidden = done;
      el.done.hidden = !done;

      // Le bouton porte l'état dans `aria-pressed` ; la CSS y accroche l'icône et
      // l'`aria-label` le dit à qui ne voit pas l'icône.
      el.play.setAttribute('aria-pressed', String(deck.playing));
      el.play.setAttribute('aria-label', deck.playing
        ? _t('viewer.pause', 'Pause')
        : _t('viewer.play', 'Lecture'));

      // Une région live qui change toutes les 1,1 s est inutilisable au lecteur d'écran :
      // l'annonce ne part que lorsque l'utilisateur a la main.
      if (!deck.playing) {
        el.live.textContent = stage.n + '. ' + _t(labelKey(stage), stage.key);
      }

      schedule();
    }

    function shapeText(stage) {
      var note = shapeNote(stage);
      if (!note) return formatDimensions(stage);
      return _t('viewer.shape_capped', 'montrée en {d} · originale {o}')
        .replace('{d}', note.displayed)
        .replace('{o}', note.original);
    }

    /** L'étape 13 : une barre par classe, triée. La sortie est une distribution. */
    function renderBars(stage) {
      el.bars.textContent = '';
      var sorted = (stage.bars || []).slice().sort(function (a, b) { return b.p - a.p; });
      sorted.forEach(function (bar) {
        var pct = adapter.truncatePercent(bar.p);
        var row = document.createElement('div');
        row.className = 'pv-bar-row';

        var label = document.createElement('span');
        label.className = 'pv-bar-label';
        label.textContent = bar.freq_hz + ' Hz';

        var track = document.createElement('div');
        track.className = 'pv-bar-track';
        var fill = document.createElement('div');
        fill.className = 'pv-bar-fill';
        fill.style.width = pct + '%';
        track.appendChild(fill);

        var value = document.createElement('span');
        value.className = 'pv-bar-value';
        value.textContent = pct.toFixed(1) + ' %';

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(value);
        el.bars.appendChild(row);
      });
    }

    function renderRail() {
      if (el.rail.childElementCount !== stages.length) buildRail();
      Array.prototype.forEach.call(el.rail.children, function (button, i) {
        var current = i === deck.index;
        button.classList.toggle('active', current);
        // Les étapes déjà franchies restent allumées, celles à venir sont estompées :
        // la bande dit d'un coup d'œil où en est le film (maquette 1a).
        button.classList.toggle('seen', i <= deck.index);
        button.setAttribute('aria-current', current ? 'true' : 'false');
        if (current) centre(button);
      });
    }

    /**
     * Ramène la tarjeta courante au milieu de la file.
     *
     * Sans cela le film avance hors de l'écran : treize étapes ne tiennent pas dans la
     * largeur du panneau, et à la sixième la tarjeta active est déjà passée à droite.
     * `block: 'nearest'` garde le mouvement horizontal : c'est la file qui défile, pas
     * le panneau qui saute.
     */
    function centre(card) {
      if (!card.scrollIntoView) return;
      card.scrollIntoView({
        behavior: prefersStill() ? 'auto' : 'smooth',
        inline: 'center',
        block: 'nearest',
      });
    }

    /**
     * La bande : une vignette par étape, dans l'ordre, chacune portant son rang.
     *
     * La maquette n'écrit que le rang sous la vignette — le nom de l'étape est déjà en
     * grand au-dessus, et treize noms sur une bande qui défile ne se lisent de toute
     * façon pas. Il reste l'`aria-label` du bouton : la bande n'est donc pas muette au
     * lecteur d'écran, et le `title` le rend au survol.
     *
     * Ce nom est écrit ici plutôt que par `data-i18n`, donc il ne suivrait pas un
     * changement de langue en cours de route — mais `show()` et l'écouteur de
     * `wittenberg:langchange` vident tous deux la bande, ce qui force cette
     * reconstruction et règle la question.
     */
    function buildRail() {
      el.rail.textContent = '';
      stages.forEach(function (stage, i) {
        var card = document.createElement('button');
        card.type = 'button';
        card.className = 'pv-step';

        var name = _t(labelKey(stage), stage.key);
        card.setAttribute('aria-label', stage.n + '. ' + name);
        card.title = name;

        var frame = document.createElement('span');
        frame.className = 'pv-step-frame';
        if (stage.png_base64) {
          var img = document.createElement('img');
          img.src = IMAGE_PREFIX + stage.png_base64;
          img.alt = '';
          if (isPixelated(stage)) img.style.imageRendering = 'pixelated';
          frame.appendChild(img);
        }
        card.appendChild(frame);

        var rank = document.createElement('span');
        rank.className = 'pv-step-n';
        rank.textContent = formatRank(stage.n);
        card.appendChild(rank);

        card.addEventListener('click', function () { deck.go(i); render(); });
        el.rail.appendChild(card);
      });
    }

    el.play.addEventListener('click', function () { deck.toggle(); render(); });
    el.prev.addEventListener('click', function () { deck.prev(); render(); });
    el.next.addEventListener('click', function () { deck.next({ manual: true }); render(); });
    el.close.addEventListener('click', function () { root.close(); });
    // « Voir le résultat » ne fait que refermer : le résultat est déjà affiché dessous,
    // et le panneau ne l'a jamais remplacé, seulement recouvert.
    el.done.addEventListener('click', function () { root.close(); });

    /**
     * Le sélecteur de langue vit dans l'en-tête du panneau (maquette 1a), donc la langue
     * peut changer film ouvert. `applyTranslations` ne touche que les nœuds `data-i18n`,
     * et tout ce que le visualiseur écrit — le nom de l'étape, sa légende, la bande,
     * l'avancement — est écrit par du JS et resterait dans l'ancienne langue.
     *
     * `WI18n.setLang` traduit, synchronise les boutons, *puis* émet cet événement : le
     * repeint arrive donc après, et c'est lui qui a le dernier mot sur l'`aria-label`
     * du bouton de lecture, que `data-i18n-aria` vient de remettre à « Lecture ».
     */
    document.addEventListener('wittenberg:langchange', function () {
      if (!root.open) return;
      el.rail.textContent = '';  // force la reconstruction : les noms y sont figés
      render();
    });

    // Échap referme le `<dialog>` sans passer par le bouton : le minuteur s'arrête ici,
    // à l'endroit où les deux chemins de fermeture se rejoignent.
    root.addEventListener('close', function () { clearTimer(); deck.pause(); });

    el.compareBtn.addEventListener('click', function () {
      var on = el.compare.hidden;
      el.compare.hidden = !on;
      el.compareBtn.setAttribute('aria-pressed', String(on));
      root.classList.toggle('pv-comparing', on);
    });

    document.addEventListener('keydown', function (event) {
      if (!root.open) return;
      var tag = event.target && event.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (event.target && event.target.isContentEditable) return;

      if (event.key === 'ArrowRight') { deck.next({ manual: true }); render(); }
      else if (event.key === 'ArrowLeft') { deck.prev(); render(); }
      else if (event.key === ' ' || event.key === 'Spacebar') {
        // Un bouton a déjà l'espace pour l'activer. Le traiter ici aussi basculerait deux
        // fois la lecture quand le focus se trouve sur « Lecture ».
        if (tag === 'BUTTON') return;
        deck.toggle();
        render();
      }
      else return;
      event.preventDefault();
    });

    return {
      show: function (list) {
        stages = (list || []).slice();
        clearTimer();
        if (stages.length < 2) return;

        el.rail.textContent = '';
        el.compare.hidden = true;
        el.compareBtn.setAttribute('aria-pressed', 'false');
        root.classList.remove('pv-comparing');
        el.compareImg.src = IMAGE_PREFIX + stages[0].png_base64;

        deck = playback(stages.length);
        // `prefers-reduced-motion` : le film ne démarre pas seul, mais rien n'est perdu —
        // la file et les flèches donnent accès aux treize étapes.
        if (!prefersStill()) deck.play();

        if (!root.open) root.showModal();
        render();
      },

      hide: function () {
        clearTimer();
        deck.pause();
        if (root.open) root.close();
      },
    };
  }

  return {
    DWELL_MS: DWELL_MS,
    LINGER_MS: LINGER_MS,
    labelKey: labelKey,
    captionKey: captionKey,
    formatMs: formatMs,
    formatShape: formatShape,
    shapeNote: shapeNote,
    formatParams: formatParams,
    formatParamsLine: formatParamsLine,
    formatDimensions: formatDimensions,
    isPixelated: isPixelated,
    edgeMapSrc: edgeMapSrc,
    dwellMs: dwellMs,
    formatRank: formatRank,
    progressPercent: progressPercent,
    isFinished: isFinished,
    playback: playback,
    mount: mount,
  };
});

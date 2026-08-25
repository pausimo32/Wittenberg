/**
 * Catalan UI copy. `CLAUDE.md` rule 6: UI strings are Catalan, code is English.
 *
 * Every physics term here resolves through `docs/07-glossary-ca-en.md` — *node*, *línia
 * nodal*, *freqüència*, *ressonància*, *escala de grisos*, *detecció de contorns*,
 * *llindar*, *exactitud*, *confiança*. Nothing is invented: if a term is missing from the
 * glossary it gets added there first, not here.
 *
 * The keys under `analyze.result` are the fixed strings the glossary's microcopy table
 * pins, reproduced exactly. Do not paraphrase them — §4 of `docs/05` treats the wording of
 * the three confidence tiers as part of the honesty requirement, not as decoration.
 */

export const t = {
  app: {
    name: 'Wittenberg',
    tagline: 'Endevina la freqüència d’una placa de Chladni a partir d’una fotografia.',
    skipToContent: 'Vés al contingut',
  },

  nav: {
    home: 'Inici',
    analyze: 'Analitza',
    learn: 'Aprèn',
    gallery: 'Galeria',
    data: 'Dades',
    about: 'Sobre',
  },

  home: {
    // §4: "one sentence, one button".
    sentence:
      'Fotografia la sorra sobre la placa i el sistema et dirà a quina freqüència vibra.',
    cta: 'Analitza una imatge',
  },

  analyze: {
    cta: 'Analitza una imatge',
    drop: 'Arrossega una fotografia aquí, o tria un fitxer',
    choose: 'Tria un fitxer',
    again: 'Analitza’n una altra',
    working: 'Analitzant…',
    privacy: 'La imatge no surt del teu dispositiu. Tot el càlcul es fa aquí.',
    original: 'Fotografia original',
    edges: 'El que veu el model',
    elapsed: 'Temps total',
    result: {
      // From the glossary's microcopy table, verbatim.
      high: 'Probablement',
      mid: 'Podria ser',
      or: 'o',
      low: 'No ho sé prou',
      lowHint:
        'Prova-ho amb més sorra, espera uns segons més o comprova que la freqüència sigui de ressonància.',
      alternatives: 'Altres candidats',
      distribution: 'La sortida del model és una distribució, no un veredicte.',
    },
    geometry: {
      // §4, "The geometry warning". Its own visible state, separate from the tier.
      title: 'No s’han detectat els quatre marcadors',
      body:
        'La placa no s’ha pogut rectificar, s’ha retallat pel centre. El model llegeix una geometria que no va veure mai durant l’entrenament, així que aquest resultat no és comparable amb els del conjunt de prova.',
      action: 'Repeteix la fotografia amb el marc de calibratge dins l’enquadrament.',
      markers: (n: number) =>
        n === 0 ? 'Cap marcador detectat' : `${n} marcadors de 4 detectats`,
    },
    error: {
      unsupported: 'No s’ha pogut llegir la imatge. Prova amb un JPEG, un PNG o un HEIC.',
      empty: 'El fitxer és buit.',
      network: 'No s’ha pogut contactar amb el servidor local.',
      model: 'No s’ha pogut carregar el model.',
    },
  },

  viewer: {
    title: 'Què ha fet la màquina amb la teva foto',
    lead:
      'Tretze passos, del que va entrar la càmera fins a la resposta. Els temps són els reals d’aquesta petició.',
    play: 'Reprodueix',
    pause: 'Atura',
    previous: 'Pas anterior',
    next: 'Pas següent',
    step: (n: number, total: number) => `Pas ${n} de ${total}`,
    compare: 'Compara amb l’original',
    keyboard: 'Fletxes per moure’t, espai per reproduir o aturar.',
    call: 'Crida',
    params: 'Paràmetres',
    size: 'Mida',
    shownAt: (w: number, h: number) => `mostrada a ${w}×${h}`,
    originalSize: (w: number, h: number) => `original ${w}×${h}`,
    groups: {
      preparation: 'Preparació',
      network: 'Dins la xarxa',
      attention: 'On ha mirat',
      decision: 'Decisió',
    },
    featureNote:
      'Cada quadre és un canal del bloc. Els primers responen a canvis simples de lluminositat; els següents, a combinacions. No sabem què detecta cap canal concret.',
    gradcamNote:
      'Les zones càlides són les que més han mogut la resposta. Quan el model s’equivoca, sovint aquí es veu per què.',
  },

  banner: {
    doNotReport: 'Aquest model no serveix per treure’n conclusions',
    run: 'Execució',
  },

  learn: { cta: 'Comença', soon: 'El mòdul didàctic arriba a la fase F7.' },
  gallery: { soon: 'La galeria de modes arriba quan hi hagi captures reals.' },
  about: { soon: 'Mètode, límits, equip i referència del TDR.' },

  data: {
    title: 'Dades',
    warning:
      'Superfície de treball de l’equip. No s’ha de mostrar mai durant una sessió de camp.',
    lead: 'Aquí s’introdueix el conjunt de dades etiquetat: una sessió de captura cada vegada.',
    newSession: 'Nova sessió',

    drafts: {
      title: 'Esborranys oberts',
      none: 'No hi ha cap esborrany obert.',
      open: 'Continua',
      discard: 'Descarta',
      discardConfirm: (id: string) =>
        `Segur que vols descartar l’esborrany ${id}? S’esborraran les fotografies que has pujat. El conjunt de dades no es tocarà.`,
      path: 'Carpeta',
      updated: 'Últim canvi',
      untouched: 'Res no s’ha escrit encara al conjunt de dades.',
      // The machine tokens the server keeps (`new` / `calibrated` / `committing`), said in
      // Catalan. A view in Catalan that prints `calibrated` has just stopped being in Catalan.
      statuses: {
        new: 'Començada',
        calibrated: 'Calibrada',
        committing: 'Incorporant-se',
      },
      counts: (accepted: number, rejected: number, duplicates: number) =>
        duplicates > 0
          ? `${accepted} acceptades · ${rejected} rebutjades · ${duplicates} duplicades`
          : `${accepted} acceptades · ${rejected} rebutjades`,
      pending: (n: number) => `${n} pendents de revisar`,
      size: (bytes: number) => {
        const units = ['B', 'kB', 'MB', 'GB', 'TB']
        let value = bytes
        let unit = 0
        while (value >= 1000 && unit < units.length - 1) {
          value /= 1000
          unit += 1
        }
        return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
      },
      interrupted:
        'La incorporació d’aquest esborrany es va interrompre. Les captures estan desades a part i encara no hi ha cap sessió: torna a incorporar-la per reprendre-ho, o descarta l’esborrany i s’esborraran amb ell.',
    },

    error: {
      // Namespaced here rather than borrowed from `analyze`: these are different failures with
      // different things to do about them, and the one that matters is `notFound` — after a
      // discard in another tab, opening an esborrany's link is a 404, not a network problem,
      // and telling someone the server is unreachable sends them to debug the wrong thing.
      list: 'No s’han pogut llegir els esborranys oberts.',
      load: 'No s’ha pogut llegir aquest esborrany.',
      notFound:
        'Aquest esborrany ja no existeix. Probablement s’ha descartat o s’ha incorporat des d’una altra pestanya.',
      discard: 'No s’ha pogut descartar l’esborrany.',
    },

    steps: {
      session: '1 · La sessió',
      calibration: '2 · El calibratge',
      frames: '3 · Les classes i les fotografies',
      review: '4 · Revisió i incorporació',
    },

    session: {
      title: 'Nova sessió de captura',
      lead:
        'Aquestes dades no es poden reconstruir a partir de les fotografies quatre dies més tard. S’omplen abans de la primera captura.',
      id: 'Identificador de la sessió',
      idHint: 'AAAAMMDD-NN, per exemple 20260830-01',
      date: 'Data',
      operator: 'Qui ha fet la captura',
      plate: 'Placa',
      material: 'Material granular',
      materials: { sand: 'sorra', semolina: 'sèmola', salt: 'sal' },
      grainMass: 'Massa de gra (g)',
      distance: 'Distància de la càmera (mm)',
      lighting: 'Il·luminació',
      lightings: {
        lamp_diffuse: 'llum difusa',
        lamp_grazing: 'llum rasant',
        ambient: 'llum ambient',
      },
      dwell: 'Temps d’estabilització (s)',
      notes: 'Notes',
      notesHint: 'Pot quedar buit. Buit vol dir “res anòmal”, que ja és informació.',
      submit: 'Comença la sessió',
    },

    calibration: {
      title: 'Calibratge',
      lead:
        'Fotografia el marc amb els quatre marcadors. Sense els quatre no hi ha placa rectificada, i sense placa rectificada la porta de qualitat no té res a mesurar.',
      drop: 'Arrossega aquí el fotograma de calibratge',
      found: (n: number) => `${n} marcadors de 4 detectats`,
      ok: 'Calibratge fet. La placa detectada és la del contorn verd.',
      failed:
        'No s’han trobat els quatre marcadors. Comprova que el marc sencer entri dins l’enquadrament i repeteix la fotografia.',
      retry: 'Repeteix el calibratge',
      pxPerMm: (v: number) => `${v.toFixed(1)} px/mm`,
      flags: 'Avisos del muntatge',
      locked: 'Primer el calibratge. Les fotografies de patrons vénen després.',
    },

    frames: {
      title: 'Classes i fotografies',
      lead: 'L’etiqueta s’escriu una vegada per classe, mai per fotografia.',
      freq: 'Freqüència (Hz)',
      addClass: 'Afegeix la classe',
      drop: 'Arrossega aquí les fotografies d’aquesta classe',
      uploading: (done: number, total: number) => `${done} de ${total}`,
      accepted: 'Acceptades',
      rejected: 'Rebutjades',
      duplicated: 'Duplicades',
      pending: 'Pendents',
      reasons: {
        blurred: 'borrosa',
        partial: 'marcadors no trobats',
        luminance: 'luminància fora de rang',
        edge_density: 'densitat de contorns insuficient',
        unreadable: 'el fitxer no es pot llegir',
      },
      original: 'Original',
      edges: 'Contorns',
      remove: 'Treu-la',
      duplicate:
        'Aquesta fotografia ja hi és. Sembla que la carpeta s’ha deixat anar dues vegades.',
      duplicateContent:
        'Aquesta captura ja és a l’esborrany amb un altre nom. Una fotografia repetida no és una segona observació, així que no s’ha desat.',
      unsupported: 'Format no admès. Han de ser HEIC, JPEG o PNG.',
      keepGoing: 'Pots continuar treballant mentre es processen.',
    },

    review: {
      title: 'Revisió',
      lead: 'Fins aquí no s’ha escrit res al conjunt de dades.',
      total: 'Captures',
      yield: 'Rendiment',
      byClass: 'Per classe',
      commit: 'Incorpora al conjunt de dades',
      committing: (done: number, total: number) =>
        `Incorporant… ${done} de ${total} fotografies`,
      staging: 'Desant les captures a part…',
      renaming: 'Registrant la sessió…',
      duplicatesNote: (n: number) =>
        `${n} ${n === 1 ? 'fotografia duplicada' : 'fotografies duplicades'} no s’han desat.`,
      done: 'Sessió incorporada.',
      failed: 'La incorporació ha fallat.',
      exists: (id: string, rows: number) =>
        `La sessió ${id} ja té ${rows} files al conjunt de dades. Incorporar-la ara les substituirà totes, i també les captures de data/raw/${id}.`,
      replace: 'Substitueix aquella sessió',
      mixedProvenance:
        'Aquest conjunt de dades barreja sessions reals i sintètiques, i això no es pot fer: una sessió SYNTH-* existeix per provar el codi, no per entrenar-hi.',
      split: 'Refés la partició',
      splitDone: (train: number, val: number, test: number) =>
        `Partició feta: ${train} entrenament · ${val} validació · ${test} prova.`,
      splitFailed:
        'No es pot partir el conjunt de dades encara. Calen almenys tres sessions perquè cada partició tingui la seva.',
      relabelWarning:
        'Si canvies una etiqueta després d’haver incorporat la sessió, cal refer la partició.',
    },

    matrix: {
      title: 'Fotografies per classe i sessió',
      lead: 'Les caselles buides són les que trenquen la partició.',
      empty: 'Encara no hi ha cap sessió incorporada.',
      sessionsShort: (freq: number, n: number, needed: number) =>
        `${freq} Hz només apareix en ${n} ${n === 1 ? 'sessió' : 'sessions'}; en calen ${needed}.`,
      cell: (accepted: number, total: number) => `${accepted}/${total}`,
    },
  },

  common: {
    close: 'Tanca',
    cancel: 'Cancel·la',
    retry: 'Torna-ho a provar',
    loading: 'Carregant…',
  },
} as const

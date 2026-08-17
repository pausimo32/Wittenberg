/**
 * frequencies.js
 * Wittenberg — Classes de fréquences expérimentales (prototypes)
 *
 * Ces valeurs sont des exemples à remplacer par les fréquences
 * réellement utilisées dans l'expérience physique.
 * Chaque classe correspond à une fréquence appliquée sur la plaque
 * et au motif de Chladni qu'elle produit.
 *
 * IMPORTANT — ce fichier n'est PAS la source de vérité des classes du modèle.
 * Les fréquences prédites proviennent exclusivement du manifeste chargé par le
 * backend (`backend/models/model_manifest.json`). Ces données ne servent qu'à
 * deux usages éditoriaux :
 *   - alimenter le tableau descriptif de `donnees.html` ;
 *   - enrichir un résultat d'analyse d'un mode vibratoire et d'une description,
 *     uniquement lorsque la fréquence prédite figure ici à l'identique.
 * Une fréquence absente de cette liste s'affiche sans description : jamais avec
 * une description approchée, qui serait scientifiquement trompeuse.
 */

const FREQUENCIES = [
  {
    id: 'f330',
    hz: 330,
    label: '330 Hz',
    mode: '(1,2)',
    description: 'Motif simple en croix. Les grains forment deux bandes perpendiculaires sur la plaque carrée.',
    nodeCount: 4,
    complexity: 'faible',
    color: '#205f75'
  },
  {
    id: 'f440',
    hz: 440,
    label: '440 Hz',
    mode: '(2,2)',
    description: 'Motif en X diagonal. Quatre zones de vibration symétriques.',
    nodeCount: 5,
    complexity: 'faible',
    color: '#2a7a96'
  },
  {
    id: 'f587',
    hz: 587,
    label: '587 Hz',
    mode: '(1,3)',
    description: 'Réseau de lignes parallèles et transversales. Structure rectangulaire marquée.',
    nodeCount: 6,
    complexity: 'moyenne',
    color: '#1d5468'
  },
  {
    id: 'f660',
    hz: 660,
    label: '660 Hz',
    mode: '(2,3)',
    description: 'Motif en étoile à 6 branches. Complexité intermédiaire.',
    nodeCount: 8,
    complexity: 'moyenne',
    color: '#205f75'
  },
  {
    id: 'f880',
    hz: 880,
    label: '880 Hz',
    mode: '(3,3)',
    description: 'Grille à 9 zones. Motif de résonance carrée classique.',
    nodeCount: 10,
    complexity: 'moyenne',
    color: '#2a7a96'
  },
  {
    id: 'f1100',
    hz: 1100,
    label: '1 100 Hz',
    mode: '(2,4)',
    description: 'Structure asymétrique complexe avec des lignes courbées caractéristiques.',
    nodeCount: 12,
    complexity: 'élevée',
    color: '#1d5468'
  },
  {
    id: 'f1320',
    hz: 1320,
    label: '1 320 Hz',
    mode: '(3,4)',
    description: 'Motif dense à 12 nœuds. Les grains forment des îlots isolés.',
    nodeCount: 14,
    complexity: 'élevée',
    color: '#205f75'
  },
  {
    id: 'f1760',
    hz: 1760,
    label: '1 760 Hz',
    mode: '(4,4)',
    description: 'Grille à 16 zones. Symétrie remarquable à haute fréquence.',
    nodeCount: 16,
    complexity: 'élevée',
    color: '#2a7a96'
  },
  {
    id: 'f2200',
    hz: 2200,
    label: '2 200 Hz',
    mode: '(3,5)',
    description: 'Structure ramifiée complexe. Forte densité de lignes nodales.',
    nodeCount: 18,
    complexity: 'très élevée',
    color: '#1d5468'
  },
  {
    id: 'f2640',
    hz: 2640,
    label: '2 640 Hz',
    mode: '(4,5)',
    description: 'Réseau très dense. Difficile à distinguer des modes voisins.',
    nodeCount: 20,
    complexity: 'très élevée',
    color: '#205f75'
  },
  {
    id: 'f3080',
    hz: 3080,
    label: '3 080 Hz',
    mode: '(5,5)',
    description: 'Motif à haute symétrie avec 25 zones. Très complexe visuellement.',
    nodeCount: 22,
    complexity: 'très élevée',
    color: '#2a7a96'
  },
  {
    id: 'f3520',
    hz: 3520,
    label: '3 520 Hz',
    mode: '(4,6)',
    description: 'Structure asymétrique à haute fréquence. Identification difficile.',
    nodeCount: 24,
    complexity: 'extrême',
    color: '#1d5468'
  },
  {
    id: 'f3910',
    hz: 3910,
    label: '3 910 Hz',
    mode: '(5,6)',
    description: 'Motif le plus complexe de la base de données. Lignes très fines.',
    nodeCount: 28,
    complexity: 'extrême',
    color: '#205f75'
  }
];

const COMPLEXITY_COLORS = {
  'faible':      '#16a34a',
  'moyenne':     '#2a7a96',
  'élevée':      '#b45309',
  'très élevée': '#c2410c',
  'extrême':     '#991b1b'
};

function getFrequencyById(id) {
  return FREQUENCIES.find(f => f.id === id) || null;
}

function getFrequencyByHz(hz) {
  return FREQUENCIES.find(f => f.hz === hz) || null;
}

function getComplexityColor(complexity) {
  return COMPLEXITY_COLORS[complexity] || '#4a6370';
}

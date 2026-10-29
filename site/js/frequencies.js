/**
 * frequencies.js
 * Wittenberg — notices éditoriales des classes de fréquences
 *
 * Vide tant que les classes réelles ne sont pas mesurées. Les treize fréquences
 * d'exemple qui vivaient ici (330 — 3 910 Hz) venaient de l'étude de référence,
 * faite sur une autre plaque : docs/02-dataset-spec.md §3 interdit de les reprendre.
 *
 * Ce fichier n'est PAS la source de vérité des classes du modèle. La liste des
 * classes vient de js/classes.generated.js, écrit par `python tasks.py modes`
 * depuis config/classes.json. Ici ne vivent que des notices, indexées par
 * fréquence, ajoutées une fois les fréquences réelles connues :
 *   { id: 'f<hz>', hz, label, mode: '(m,n)', description, nodeCount, complexity, color }
 * Elles servent à deux usages éditoriaux :
 *   - alimenter le tableau descriptif de `donnees.html` ;
 *   - enrichir un résultat d'analyse d'un mode vibratoire et d'une description,
 *     uniquement lorsque la fréquence prédite figure ici à l'identique.
 * Une fréquence absente de cette liste s'affiche sans description : jamais avec
 * une description approchée, qui serait scientifiquement trompeuse.
 */

const FREQUENCIES = [];

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

# Modèles et manifestes

Ce dossier reçoit l'artefact du modèle et son manifeste. **Aucun modèle
entraîné n'est fourni avec le dépôt** : tant qu'il est absent, le service
démarre mais `/ready` répond `503` et l'analyseur reste désactivé.

## Artefacts attendus

| Fichier | Rôle | Présent ? |
| --- | --- | --- |
| `model_manifest.json` | Manifeste versionné : classes, géométrie d'entrée, prétraitement | **manquant** |
| `chladni_cnn.onnx` (ou `.keras` / `.h5` / `.tflite`) | Poids du modèle entraîné | **manquant** |
| `model_manifest.example.json` | Gabarit à copier | fourni |

Le nom de l'artefact est libre : c'est le champ `model.artifact` du manifeste
qui fait foi, résolu relativement au dossier du manifeste.

## Politique de versionnement des binaires

Les poids d'un CNN pèsent typiquement de quelques mégaoctets à plusieurs
centaines. Ce dépôt **n'utilise ni Git LFS ni de stockage binaire** : il ne
contient aujourd'hui que du HTML, du CSS, du JavaScript et des polices.

**Ne committez pas l'artefact directement.** Choisissez l'une des options :

1. **Volume monté** (recommandé pour Docker) :
   `docker run -v /chemin/vers/modeles:/models …`
2. **Copie locale hors dépôt**, désignée par `WITTENBERG_MODEL_PATH`.
3. **Téléchargement au déploiement** depuis un stockage externe, avec
   vérification du SHA-256 déclaré dans le manifeste.
4. **Git LFS**, uniquement après avoir activé `git lfs` sur le dépôt et
   documenté la décision — le `.gitignore` racine exclut actuellement
   `*.onnx`, `*.keras`, `*.h5` et `*.tflite`.

## Mettre un modèle en service

```bash
# 1. Copier le gabarit
cp models/model_manifest.example.json models/model_manifest.json

# 2. Renseigner : nom, version, artefact, géométrie d'entrée, prétraitement,
#    et surtout la correspondance exacte index -> fréquence.

# 3. Déposer l'artefact à côté du manifeste, puis inscrire son empreinte
python -m wittenberg.tools.validate_manifest \
    --manifest models/model_manifest.json --write-hash

# 4. Vérifier que manifeste et modèle sont cohérents (charge et préchauffe)
python -m wittenberg.tools.validate_manifest --manifest models/model_manifest.json
```

La dernière commande échoue si le nombre de sorties, la géométrie d'entrée ou
l'empreinte ne concordent pas. Elle doit passer **avant** tout déploiement.

## L'ordre des classes

L'index de chaque classe est déclaré explicitement dans le manifeste :

```json
{ "index": 0, "label": "440 Hz", "frequency_hz": 440.0 }
```

Cet ordre doit reproduire **exactement** celui utilisé à l'entraînement.

> Avec Keras, `image_dataset_from_directory` trie les dossiers par ordre
> alphabétique : `"1000Hz"` arrive alors **avant** `"440Hz"`. Ne recopiez
> jamais un ordre supposé — relevez `dataset.class_names` à l'entraînement et
> reportez-le tel quel dans le manifeste. Le service refuse tout index
> dupliqué ou manquant, mais il ne peut pas deviner une permutation erronée :
> le modèle prédirait alors systématiquement la mauvaise fréquence.

## Choix du format

| Format | Runtime | Python | Remarque |
| --- | --- | --- | --- |
| **ONNX** | `onnxruntime` | 3.11 → 3.14 | **Recommandé.** Léger, réentrant, sans TensorFlow |
| Keras (`.keras`, `.h5`, SavedModel) | `tensorflow` | 3.11 → 3.13 | **Aucune roue pour Python 3.14** |
| TFLite | `tflite-runtime` ou `tensorflow` | 3.11 → 3.13 | Très léger, adapté à un petit modèle |

Exporter un modèle Keras entraîné vers ONNX :

```bash
pip install tf2onnx
python -m tf2onnx.convert --keras chladni_cnn.keras --output chladni_cnn.onnx --opset 13
```

Puis, dans le manifeste : `"framework": "onnx"`, `"format": "onnx"`.

## Vérifier le prétraitement

Avant d'entraîner, contrôlez ce que le modèle recevra réellement :

```bash
python -m wittenberg.tools.preprocess_cli photo.jpg \
    --manifest models/model_manifest.json --save-preview vue.png
```

Le même code produit le tenseur à l'entraînement, à l'évaluation et à
l'inférence — c'est ce qui garantit qu'ils ne divergent pas.

## Sections du manifeste

| Section | Contenu |
| --- | --- |
| `model` | nom, version, framework, format, artefact, `sha256`, date d'entraînement |
| `input` | `height`, `width`, `channels`, `layout` (NHWC/NCHW), `dtype`, `value_range`, `normalization` |
| `preprocessing` | `version`, gris, détection de plaque, contraste, débruitage, Canny, redimensionnement |
| `output` | `activation` : `softmax` ou `logits` |
| `classes` | `index`, `label`, `frequency_hz` — l'ordre de référence |
| `decision` | `min_confidence`, `min_margin`, `top_k` |
| `dataset` | identifiant, date, nombre d'images par classe, notes |

Toute modification du comportement de prétraitement impose d'incrémenter
`preprocessing.version`, et tout réentraînement d'incrémenter `model.version`.

# Wittenberg — service d'inférence

API HTTP qui reçoit la photographie d'un motif de Chladni, la valide, la
prétraite de façon déterministe, exécute le CNN et renvoie la fréquence
estimée, sa confiance et les hypothèses alternatives.

> **État actuel — aucun modèle n'est fourni.**
> Le service démarre, mais `/ready` répond `503` et `/api/v1/predict` renvoie
> `503 model_unavailable` tant que le manifeste et l'artefact ne sont pas
> déposés. Aucune prédiction simulée n'est jamais produite.
> Voir [`models/README.md`](models/README.md) pour les artefacts attendus.

---

## 1. Installation

Prérequis : Python 3.11 ou plus récent (3.14 accepté avec le runtime ONNX).

```bash
cd backend

python -m venv .venv
# Windows (PowerShell) :  .venv\Scripts\Activate.ps1
# Windows (Git Bash)   :  source .venv/Scripts/activate
# Linux / macOS        :  source .venv/bin/activate

pip install -r requirements.txt        # production
pip install -r requirements-dev.txt    # + tests, lint, typage
```

Puis la configuration :

```bash
cp .env.example .env
```

> **TensorFlow et Python 3.14.** Aucune roue TensorFlow n'existe pour Python
> 3.14. Pour servir un artefact `.keras`, `.h5` ou SavedModel, utilisez Python
> 3.11–3.13 et `pip install tensorflow==2.20.0`, ou — plus simple — exportez le
> modèle en ONNX. Le runtime ONNX couvre Python 3.11 à 3.14.

---

## 2. Lancer le service

```bash
# Développement — rechargement automatique
uvicorn wittenberg.app:app --reload --port 8000 --app-dir src

# Production — un seul processus suffit : le modèle est chargé par processus
uvicorn wittenberg.app:app --host 0.0.0.0 --port 8000 --workers 1 --app-dir src
```

Documentation interactive : <http://localhost:8000/docs>

Si le paquet est installé (`pip install -e .`), `--app-dir src` devient inutile.

### Docker

```bash
docker build -t wittenberg-backend ./backend

docker run --rm -p 8000:8000 \
  -v "$(pwd)/backend/models:/models:ro" \
  -e WITTENBERG_ALLOWED_ORIGINS=http://localhost:5500 \
  wittenberg-backend
```

L'artefact n'est jamais copié dans l'image : il est monté en lecture seule.

---

## 3. Tests et contrôles

```bash
python -m pytest              # 164 tests, environ 2 s
python -m ruff check .        # lint
python -m ruff format --check .
python -m mypy                # typage strict de src/
```

Les tests fabriquent leurs propres images et un petit modèle ONNX réel : le
dépôt ne contient aucun binaire de test. Un adaptateur factice existe
uniquement dans `tests/` ; la production ne peut pas démarrer dessus.

---

## 4. API

Toutes les erreurs partagent la même forme, sans trace interne :

```json
{ "request_id": "…", "error": { "code": "corrupt_image", "message": "…" } }
```

Chaque réponse porte un en-tête `X-Request-ID`.

### `GET /health`

Vivacité du processus. Ne dit rien du modèle.

```json
{ "status": "ok", "service": "wittenberg-backend", "version": "0.1.0" }
```

### `GET /ready`

`200` si un modèle est chargé, validé et préchauffé ; `503` sinon.

```json
{
  "ready": false,
  "status": "not_configured",
  "reason": "Aucun modèle n'est configuré sur ce serveur.",
  "missing": ["WITTENBERG_MODEL_MANIFEST_PATH n'est pas définie"]
}
```

### `GET /api/v1/model-info`

Toujours `200`, afin que le frontend puisse afficher un état précis. Expose la
version, les classes dans l'ordre du manifeste et les contraintes d'entrée.

```json
{
  "available": true,
  "status": "ready",
  "model": { "name": "wittenberg-chladni-cnn", "version": "1.0.0" },
  "framework": "onnx",
  "runtime": "onnxruntime 1.28.0",
  "preprocessing_version": "prep-1.0.0",
  "input": { "height": 224, "width": 224, "channels": 1, "layout": "NHWC",
             "dtype": "float32", "value_range": [0.0, 1.0], "normalization": "scale_01" },
  "num_classes": 13,
  "classes": [{ "index": 0, "label": "…", "frequency_hz": 0 }],
  "min_confidence": 0.55,
  "requirements": { "max_upload_mb": 10.0, "accepted_formats": ["bmp", "jpeg", "png", "tiff", "webp"],
                    "field_name": "image" }
}
```

### `POST /api/v1/predict`

`multipart/form-data`, un seul fichier dans le champ **`image`** (`file` est
accepté comme alias).

```bash
curl -X POST http://localhost:8000/api/v1/predict \
     -F "image=@photo_plaque.jpg"
```

```json
{
  "request_id": "8f3a…",
  "prediction": {
    "label": "440 Hz",
    "frequency_hz": 440.0,
    "class_index": 1,
    "confidence": 0.9533,
    "reliable": true,
    "reason": null,
    "margin": 0.8712
  },
  "top_k": [
    { "label": "440 Hz", "frequency_hz": 440.0, "class_index": 1, "probability": 0.9533 },
    { "label": "587 Hz", "frequency_hz": 587.0, "class_index": 2, "probability": 0.0821 }
  ],
  "quality": {
    "score": 0.87,
    "warnings": [{ "code": "perspective", "message": "La plaque est vue de biais…",
                   "severity": "warning" }],
    "metrics": { "sharpness": 412.5, "contrast": 58.3, "edge_density": 0.041, "coverage": 0.72 },
    "plate_detected": true,
    "plate_method": "homography"
  },
  "model": { "name": "wittenberg-chladni-cnn", "version": "1.0.0" },
  "preprocessing": { "version": "prep-1.0.0", "steps": ["plate:homography", "grayscale", "…"] },
  "processing_ms": 123.4
}
```

**Vocabulaire.** `confidence` et `probability` qualifient **une** sortie. Ce ne
sont ni une précision ni un taux de réussite : ces notions décrivent le modèle
dans son ensemble et se mesurent sur un jeu de validation (§ 7).

Une image valide mal reconnue renvoie `200` avec `reliable: false` et un
`reason` (`low_confidence` ou `ambiguous`) — jamais une erreur `4xx`.

#### Codes d'erreur

| Code HTTP | `error.code` | Cause |
| --- | --- | --- |
| 400 | `empty_file` | Fichier vide |
| 400 | `corrupt_image` | Image illisible ou tronquée |
| 400 | `image_too_small` | En dessous de `MIN_IMAGE_DIMENSION` |
| 413 | `file_too_large` | Au-delà de `MAX_UPLOAD_MB` |
| 413 | `image_too_large` / `image_too_many_pixels` | Dimensions ou pixels excessifs |
| 415 | `unsupported_format` | Contenu non reconnu, ou GIF |
| 415 | `unsupported_heif` | HEIC/HEIF sans `pillow-heif` installé |
| 422 | `missing_file` / `invalid_request` | Requête mal formée |
| 503 | `model_unavailable` | Aucun modèle chargé |
| 503 | `preprocessing_unavailable` | Prétraitement exigé mais indisponible |
| 504 | `inference_timeout` | Délai d'inférence dépassé |

```bash
# Quelques requêtes invalides, à titre de vérification
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/api/v1/predict          # 422
curl -X POST http://localhost:8000/api/v1/predict -F "image=@README.md;filename=faux.png"      # 415
```

---

## 5. Pipeline de prétraitement

Implémentation **unique** dans `src/wittenberg/imaging/pipeline.py`, partagée
par l'API, l'évaluation et l'outil en ligne de commande. Le comportement est
piloté par la section `preprocessing` du manifeste et versionné par
`preprocessing.version`.

1. **Validation** — signature binaire réelle (jamais l'extension ni le
   `Content-Type`), taille, dimensions, nombre total de pixels.
2. **Décodage sûr** — image tronquée refusée, pas de bombe de décompression.
3. **Orientation EXIF** — la photo de téléphone est redressée.
4. **Suppression des métadonnées** — EXIF, GPS et ICC sont abandonnés.
5. **Conversion en RGB** — RGBA/LA/P composées sur du noir ; CMYK, gris et
   16 bits convertis de façon déterministe.
6. **Localisation de la plaque** — seuillage d'Otsu, contours, quadrilatère
   convexe. Si le résultat est plausible, redressement par homographie vers un
   carré. Sinon **aucune homographie n'est inventée** : recadrage central
   déterministe et avertissement `plate_not_detected`.
7. **Mesures de qualité** — netteté, exposition, contraste, couverture,
   perspective, présence d'un motif.
8. **Niveaux de gris** — luminance ITU-R BT.601.
9. **Contraste** — `none`, `minmax` ou `clahe` (CLAHE exige OpenCV).
10. **Débruitage** — `none`, `median` ou `gaussian`.
11. **Redimensionnement** — `pad` conserve la géométrie du motif et complète
    les bords ; `stretch` déforme.
12. **Contours (Canny)** — appliqué **après** redimensionnement, et uniquement
    si le modèle a été entraîné ainsi. Exige OpenCV : à défaut, le service se
    déclare indisponible plutôt que de produire un tenseur différent.
13. **Tenseur** — canaux, normalisation, écrêtage sur `value_range`,
    disposition NHWC/NCHW, `dtype` du manifeste.

Le frontend n'effectue **aucun** de ces traitements : il envoie l'image
d'origine. Rien n'est donc appliqué deux fois, et le navigateur ne peut pas
diverger du prétraitement d'entraînement.

Inspecter le résultat :

```bash
python -m wittenberg.tools.preprocess_cli photo.jpg \
    --manifest models/model_manifest.json --save-preview vue.png
```

---

## 6. Variables d'environnement

Toutes préfixées par `WITTENBERG_`. Voir [`.env.example`](.env.example).

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `MODEL_MANIFEST_PATH` | *(vide)* | Manifeste. Sans lui, `/ready` répond 503 |
| `MODEL_PATH` | *(vide)* | Artefact ; sinon celui du manifeste |
| `VERIFY_MODEL_HASH` | `true` | Contrôle du SHA-256 au démarrage |
| `WARMUP_ON_STARTUP` | `true` | Inférence de préchauffage |
| `STRICT_STARTUP` | `false` | `true` : refuse de démarrer sans modèle valide |
| `MAX_UPLOAD_MB` | `10` | Taille maximale acceptée |
| `MAX_IMAGE_PIXELS` | `40000000` | Garde anti-bombe de décompression |
| `MAX_IMAGE_DIMENSION` | `12000` | Côté maximal |
| `MIN_IMAGE_DIMENSION` | `32` | Côté minimal |
| `INFERENCE_TIMEOUT_MS` | `15000` | Délai d'inférence |
| `TOP_K` | `3` | Hypothèses renvoyées |
| `ONNX_INTRA_OP_THREADS` | `1` | Fils par inférence |
| `ALLOWED_ORIGINS` | localhost:5500, :8080 | Origines CORS, séparées par des virgules. `*` est ignoré |
| `API_PREFIX` | `/api/v1` | Préfixe des routes d'inférence |
| `LOG_LEVEL` | `INFO` | Niveau de journalisation |
| `ENABLE_DEBUG_PREVIEW` | `false` | Renvoie l'image prétraitée en base64 |

---

## 7. Évaluer le modèle

```bash
python -m wittenberg.tools.evaluate \
    --manifest models/model_manifest.json \
    --data data/validation \
    --train-data data/train \
    --session-pattern 'session[-_]?(\d+)' \
    --output rapport_validation.json
```

Produit : accuracy, top-k accuracy, macro-F1, précision/rappel/F1 par classe,
matrice de confusion, effectifs par fréquence, confiance moyenne et nombre de
cas sous le seuil.

**Fuite de données.** Les clichés d'une même session expérimentale se
ressemblent fortement ; les répartir entre entraînement et validation
surestime les performances. Avec `--train-data` et `--session-pattern`, le
script signale les noms de fichiers et les sessions présents des deux côtés.

---

## 8. Sécurité et confidentialité

- Type déterminé par le **contenu réel**, jamais par l'extension ou le
  `Content-Type`.
- Limites de taille, de dimensions et de nombre de pixels appliquées **avant**
  décodage ; la lecture s'arrête au premier octet excédentaire.
- Le nom de fichier fourni par le client n'est **jamais** utilisé comme chemin.
- **Aucune image n'est stockée** ; les fichiers temporaires de téléversement
  sont refermés dans tous les cas, y compris en erreur.
- Métadonnées EXIF et GPS supprimées au décodage.
- Journaux structurés en JSON, **sans image, sans binaire, sans nom de fichier
  utilisateur** — uniquement des mesures et des identifiants techniques.
- Les messages d'erreur ne contiennent aucune trace interne.
- CORS par liste explicite ; `*` est filtré.
- **Aucun artefact provenant d'une requête n'est désérialisé** : seul le
  fichier de confiance configuré côté serveur est chargé.
- Aucune image n'est envoyée à un service externe ; aucune télémétrie ; aucune
  base de données.

---

## 9. Organisation du code

```
backend/
  src/wittenberg/
    app.py                    fabrique FastAPI, CORS, identifiant de requête
    config.py                 configuration par variables d'environnement
    logging_config.py         journalisation JSON
    api/
      routes.py               /health, /ready, /model-info, /predict
      schemas.py              contrats de réponse
      errors.py               gestionnaires d'erreurs
    imaging/
      validation.py           validation et décodage sûrs
      plate.py                localisation de la plaque, homographie
      quality.py              indicateurs de qualité
      pipeline.py             pipeline unique de prétraitement
    model/
      manifest.py             schéma et validation du manifeste
      registry.py             choix de l'adaptateur
      adapters/               ONNX, Keras, TFLite
      decision.py             probabilités, top-k, fiabilité
      service.py              chargement unique, validation, inférence
    tools/
      preprocess_cli.py       prétraitement hors API
      evaluate.py             métriques sur jeu de validation
      validate_manifest.py    contrôle manifeste ↔ modèle
  models/                     manifeste et artefact (non versionnés)
  tests/                      164 tests
```

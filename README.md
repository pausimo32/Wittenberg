# Wittenberg

Outil de recherche pour l'identification visuelle de fréquences sonores à
partir de motifs de Chladni photographiés sur une plaque vibrante.

Projet TDR — physique des ondes, vision par ordinateur et apprentissage
automatique supervisé.

## Structure

```
.                       frontend statique : HTML, CSS, JavaScript, sans build
  index.html            présentation du projet
  analyseur.html        outil principal : import ou caméra, puis analyse
  experience.html       dispositif expérimental
  donnees.html          jeu de données et classes de fréquences
  apropos.html          cadre académique du TDR
  css/ js/ assets/
backend/                service d'inférence Python (FastAPI + ONNX Runtime)
  README.md             installation, API, prétraitement, variables, tests
  models/README.md      comment fournir ou remplacer le modèle
```

## Démarrer en local

Deux serveurs : l'API d'inférence et le site statique.

**1. Backend** — voir [`backend/README.md`](backend/README.md) pour le détail.

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
# .venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
cp .env.example .env
uvicorn wittenberg.app:app --reload --port 8000 --app-dir src
```

**2. Frontend** — depuis la racine du dépôt, dans un second terminal :

```bash
python -m http.server 5500
```

Puis ouvrir <http://127.0.0.1:5500/analyseur.html>.

Le port 5500 figure déjà dans `WITTENBERG_ALLOWED_ORIGINS`. Pour un autre
port, ajoutez l'origine correspondante dans `backend/.env`.

> Ouvrir les fichiers HTML directement (`file://`) empêche le navigateur
> d'appeler l'API : passez toujours par un serveur local.

## Adresse de l'API côté navigateur

Le frontend est statique : il n'a ni build ni variables d'environnement.
`js/config.js` résout l'adresse de l'API, par ordre de priorité :

1. `<meta name="wittenberg-api-base" content="https://api.exemple.org">`
2. `window.WITTENBERG_API_BASE_URL`, définie avant le chargement du script
3. `localStorage.setItem('wittenberg.apiBaseUrl', '…')`
4. par défaut : `http://localhost:8000` en local, même origine en production

## État du projet

L'analyseur est relié à une véritable API d'inférence ; **aucune prédiction
n'est simulée**. Il manque uniquement le modèle entraîné : tant qu'il n'est
pas déposé, `/ready` répond `503`, l'analyse reste désactivée et l'interface
l'indique explicitement.

Artefacts à fournir dans `backend/models/` :

- `model_manifest.json` — classes, ordre `index → fréquence`, géométrie
  d'entrée, version du prétraitement ;
- l'artefact du modèle (`.onnx` recommandé).

Marche à suivre détaillée : [`backend/models/README.md`](backend/models/README.md).

## Documents de référence

- `WITTENBERG_CONTEXT.md` — portée scientifique et fonctionnelle
- `WITTENBERG_BRAND_GUIDE.md` — identité visuelle
- `CLAUDE.md` — consignes de contribution

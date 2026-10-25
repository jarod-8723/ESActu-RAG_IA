# 🎓 ESActu

Assistant IA conversationnel du Master ESA (Économétrie et Statistique Appliquée) — Université d'Orléans.

ESActu répond aux questions des étudiants et futurs candidats à partir du contenu officiel du site [master-esa.fr](https://www.master-esa.fr), via un pipeline RAG (Retrieval-Augmented Generation) : scraping → ingestion vectorielle → interface conversationnelle.

---

## 📐 Architecture du projet

```
ESActu - V2/
├── .venv/                     # Environnement virtuel Python
├── data/                      # Fichiers .md / .csv générés par le scraping
├── chroma_db/                 # Base vectorielle ChromaDB persistée
├── récolte-site/
│   └── scraping.py            # Scraping du site master-esa.fr → data/
├── ingestion-data/
│   └── ingest.py              # Ingestion incrémentale de data/ → chroma_db/
├── interface-esactu/
│   ├── interface.py           # Interface Chainlit (RAG + Groq)
│   ├── interface.env          # Variables d'environnement (modèle, clé API)
│   ├── system_prompt.txt      # Prompt système du LLM
│   ├── chainlit.md            # Page d'accueil Chainlit
│   ├── .chainlit/config.toml  # Configuration Chainlit (UI, features)
│   └── public/                # Logos, CSS, GIF de connexion
├── models/
│   └── bge-m3/                # Modèle d'embedding local (BAAI/bge-m3)
├── utilisateurs/
│   └── users_esia.xlsx        # Comptes utilisateurs (visiteur / membre)
└── requirements.txt
```

---

## 🔄 Pipeline

```
1. SCRAPING (récolte-site/scraping.py)
   master-esa.fr → fichiers .md classés par section (Formation, Candidature,
   Insertion, Alumni, Ressources, Contact, Équipe) dans data/

2. INGESTION (ingestion-data/ingest.py)
   data/*.md → chunking (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter)
   → embedding (bge-m3) → upsert incrémental dans chroma_db/
   Un registre JSON (ingest_registry.json) suit les fichiers déjà ingérés via
   hash SHA-256, pour ne réingérer que les fichiers nouveaux/modifiés/supprimés.

3. INTERFACE (interface-esactu/interface.py)
   Question utilisateur → recherche vectorielle (MMR, k=15) filtrée par rôle
   → construction du contexte → appel Groq (streaming) → réponse + questions
   annexes suggérées.
```

---

## 🧩 Détail des composants

### `scraping.py`

Crawler qui parcourt `master-esa.fr`, convertit chaque page HTML (et les PDF liés) en Markdown avec frontmatter YAML (`url`, `scraped_at`, `title`, `section`, `promotion`), et classe les fichiers dans `data/<Section>/`.

- Détection automatique de la section via `SECTION_MAP` (mapping slug URL → section)
- Détection des promotions Alumni par plage d'années (`Promotion_2023-2024`)
- Nettoyage du HTML (suppression nav/footer/cookies/formulaires) avant conversion
- Conversion des PDF en Markdown via `pymupdf4llm`
- Filtre les pages trop courtes (< 50 mots) pour éviter le bruit

### `ingest.py`

Pipeline d'ingestion incrémental :

- `load_documents()` — charge les `.md`/`.csv`, extrait le frontmatter
- `split_documents()` — découpe en chunks (headers Markdown puis taille de caractères)
- `build_vectorstore()` — calcule les embeddings (bge-m3) et upsert par batch dans ChromaDB
- Registre JSON (`ingest_registry.json`) : hash SHA-256 par fichier, ne réingère que le delta (nouveau / modifié / supprimé)
- `force_reingest=True` pour une réingestion complète (ex : changement de modèle d'embedding)

### `interface.py`

Interface Chainlit avec :

- **Authentification** par identifiants (fichier Excel `users_esia.xlsx`), rôles `visiteur` / `membre` (accès Alumni réservé aux membres)
- **Sélection dynamique du modèle Groq** : lit `GROQ_MODEL` depuis `.env`, si indisponible sur le compte bascule automatiquement sur un modèle Qwen disponible
- **Starters** (questions prédéfinies sur l'écran d'accueil) + réponses statiques pour certaines (contact, crédits)
- **Recherche vectorielle** MMR (k=15) filtrée selon les sections autorisées par le rôle
- **Génération en streaming** avec animation "thinking" pendant la recherche
- **Questions annexes** suggérées automatiquement, extraites de la réponse du LLM et affichées en boutons cliquables

---

## ⚙️ Configuration

### `interface.env` (dans `interface-esactu/`)

```env
CHAINLIT_AUTH_SECRET=<secret_généré>
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_LLM_API=<clé_api_groq>
```

### `system_prompt.txt` (dans `interface-esactu/`)

Prompt système du LLM, chargé au démarrage de l'interface. Définit le ton, la structure de réponse (Markdown avec emojis, flèches ➜), et impose la génération d'une section `### ❓ Questions annexes` en fin de réponse.

### `.chainlit/config.toml`

Configuration de l'UI Chainlit : nom de l'app, logos (`logo_light.png`, `logoESA.png`), thème clair forcé, langue française, CSS personnalisé (`custom.css`), GIF de connexion (`login2.gif`), rendu LaTeX activé.

---

## 🚀 Installation

```powershell
cd "ESActu - V2"
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### 1. Scraper le site

```powershell
cd récolte-site
python scraping.py
```
→ génère les fichiers `.md` dans `data/`.

### 2. Ingérer les données

```powershell
cd ingestion-data
python ingest.py
```
→ met à jour `chroma_db/` (incrémental par défaut).

### 3. Lancer l'interface

```powershell
cd interface-esactu
chainlit run interface.py
```
→ disponible sur `http://localhost:8000`.

---

## 👥 Gestion des utilisateurs

Les comptes sont définis dans `utilisateurs/users_esia.xlsx`, feuille `Utilisateurs`, à partir de la ligne 5 :

| Colonne | Contenu |
|---|---|
| D | Nom d'utilisateur |
| E | Mot de passe |
| F | Rôle (`visiteur` ou `membre`) |

Les **visiteurs** ont accès aux sections : Formation, Candidature, Insertion, Ressources, Contact, Équipe, Accueil.
Les **membres** ont accès en plus à la section Alumni.

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Scraping | BeautifulSoup, html2text, pymupdf4llm |
| Embedding | BAAI/bge-m3 (local, HuggingFace) |
| Base vectorielle | ChromaDB (persistant) |
| LLM | Groq API (modèle Qwen, sélection dynamique) |
| Interface | Chainlit |
| Authentification | Fichier Excel (openpyxl) |

---

## 👨‍💻 Auteurs

Fatoumata, Marius et Jarod — étudiants en Master ESA (Économétrie et Statistique Appliquée), Université d'Orléans, dans le cadre d'un projet académique.

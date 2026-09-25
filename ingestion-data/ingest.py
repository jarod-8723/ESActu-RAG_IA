# -*- coding: utf-8 -*-
"""
Pipeline d'ingestion RAG : transforme des fichiers Markdown et CSV en
vecteurs stockés dans ChromaDB pour être interrogés par un LLM.

Antérieurement, le site master-esa.fr a été récupéré dans un format brut
pour ensuite être converti en format .md

Le script va découper les fichiers Markdown en plusieurs chunks
Chaque chunk sera positionné dans un espace vectoriel
   => C'est la "base vectorielle"


Flux général :
  Fichiers .md / .csv
       ↓
  load_documents()       → liste de Documents LangChain (1 doc = 1 fichier)
       ↓
  split_documents()      → chunks (morceaux) plus petits, prêts pour l'embedding
       ↓
  build_vectorstore()    → calcul des embeddings + stockage ChromaDB sur disque
       ↓
  ChromaDB persisté      → interrogeable via un retriever (interface.py)
"""

import re
import csv
import json
import yaml
import hashlib
import logging
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Logging — configuration globale
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — fonctions utilitaires internes
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """
    Détecte si un GPU CUDA est disponible pour accélérer l'embedding.
    Retourne 'cuda' si PyTorch trouve un GPU, sinon 'cpu'.
    Le modèle d'embedding tourne ~5x plus vite sur GPU.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB n'accepte que les types scalaires : str, int, float, bool.
    Cette fonction nettoie les métadonnées avant le stockage :
      - Les valeurs None sont ignorées (supprimées de la dict)
      - Les listes, tuples, dicts sont convertis en str
      - Les types compatibles (str, int, float, bool) passent tels quels
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif value is None:
            pass  # On ignore plutôt que de stocker la chaîne "None"
        else:
            sanitized[key] = str(value)
    return sanitized


def _generate_chunk_ids(chunks: list[Document]) -> list[str]:
    """
    Génère un identifiant stable et unique pour chaque chunk.

    Pourquoi MD5 et pas un index séquentiel ?
    → Si un fichier source est modifié, seuls les chunks réellement changés
      auront un nouvel ID → l'upsert ChromaDB met à jour uniquement ceux-là.
    → Avec un index séquentiel, tout décalage de position créerait de faux doublons.

    Format de l'ID : "<chemin_fichier>::<12 premiers hex du MD5 du contenu>"
    Si collision (contenu identique dans le même fichier) : "...::1", "...::2"
    """
    seen: dict[str, int] = {}
    ids = []

    for chunk in chunks:
        filepath = chunk.metadata.get("filepath", "unknown")
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
        base_id = f"{filepath}::{content_hash}"

        if base_id in seen:
            seen[base_id] += 1
            final_id = f"{base_id}::{seen[base_id]}"
        else:
            seen[base_id] = 0
            final_id = base_id

        ids.append(final_id)

    return ids


# ---------------------------------------------------------------------------
# Registre de fichiers — suivi des fichiers déjà ingérés
# ---------------------------------------------------------------------------

def _file_hash(filepath: Path) -> str:
    """
    Calcule le SHA-256 du contenu d'un fichier.
    Utilisé pour détecter si un fichier a été modifié depuis la dernière ingestion.
    """
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def _load_registry(registry_path: Path) -> dict:
    """
    Charge le registre JSON des fichiers déjà ingérés.

    Structure du registre :
      {
        "chemin/absolu/fichier.md": {
          "hash": "sha256...",
          "ingested_at": "2024-...",
          "chunk_ids": ["id1", "id2"]
        },
        ...
      }

    Retourne un dict vide si le fichier n'existe pas encore (première ingestion).
    """
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Registre illisible (%s) — sera recréé.", e)
        return {}


def _save_registry(registry: dict, registry_path: Path) -> None:
    """Persiste le registre JSON sur disque (création du dossier si nécessaire)."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  ✓ Registre mis à jour : %s (%d fichiers suivis)", registry_path, len(registry))


def _compute_file_delta(
    data_dir: str,
    registry: dict,
    max_file_size_mb: float,
) -> tuple[list[Path], list[str]]:
    """
    Compare les fichiers .md et .csv présents sur disque avec le registre.

    Retourne :
      - new_or_modified : fichiers à (ré)ingérer (nouveaux ou hash différent)
      - deleted_filepaths : chemins de fichiers supprimés depuis la dernière ingestion
                            → leurs chunk_ids devront être retirés de ChromaDB
    """
    data_path = Path(data_dir)
    max_bytes = max_file_size_mb * 1024 * 1024
    current_paths: set[str] = set()
    new_or_modified: list[Path] = []

    for filepath in sorted(list(data_path.rglob("*.md")) + list(data_path.rglob("*.csv"))):
        if filepath.stat().st_size > max_bytes:
            continue

        str_path = str(filepath)
        current_paths.add(str_path)
        current_hash = _file_hash(filepath)

        if str_path not in registry:
            logger.info("    [NEW]      %s", filepath.name)
            new_or_modified.append(filepath)
        elif registry[str_path]["hash"] != current_hash:
            logger.info("    [MODIFIÉ]  %s", filepath.name)
            new_or_modified.append(filepath)
        else:
            logger.debug("    [INCHANGÉ] %s", filepath.name)

    deleted_filepaths = [p for p in registry if p not in current_paths]
    for p in deleted_filepaths:
        logger.info("    [SUPPRIMÉ] %s", Path(p).name)

    return new_or_modified, deleted_filepaths


def _purge_deleted_chunks(
    collection: chromadb.Collection,
    registry: dict,
    deleted_filepaths: list[str],
) -> None:
    """
    Supprime de ChromaDB tous les chunks appartenant à des fichiers supprimés.
    On utilise les chunk_ids mémorisés dans le registre lors de la dernière ingestion.
    """
    if not deleted_filepaths:
        return

    ids_to_delete: list[str] = []
    for path in deleted_filepaths:
        ids_to_delete.extend(registry.get(path, {}).get("chunk_ids", []))

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("  ✓ %d chunks supprimés (fichiers retirés du corpus)", len(ids_to_delete))


# ---------------------------------------------------------------------------
# Frontmatter YAML — métadonnées en tête de fichier Markdown
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Les fichiers Markdown du scraper commencent par un bloc YAML entre ---
    appelé « frontmatter » (url, date de scraping, titre, section…).

    Exemple de frontmatter :
        ---
        url: https://www.master-esa.fr/la-formation
        scraped_at: 2026-04-28
        title: "La Formation"
        section: Formation
        ---

    Cette fonction :
      1. Détecte et extrait ce bloc via une regex
      2. Le parse en dict Python (yaml.safe_load)
      3. Aplatit les listes en string CSV (ChromaDB n'accepte pas les listes)
      4. Retourne (metadata_dict, contenu_sans_frontmatter)

    Si pas de frontmatter → retourne ({}, texte_original)
    """
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}, text

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
        metadata = {
            k: ", ".join(v) if isinstance(v, list) else v
            for k, v in metadata.items()
        }
    except yaml.YAMLError:
        metadata = {}

    return metadata, text[match.end():]


# ---------------------------------------------------------------------------
# Loaders — chargement des fichiers sources en Documents LangChain
# ---------------------------------------------------------------------------

def load_csv(filepath: Path, text_columns: list[str] | None = None) -> list[Document]:
    """
    Charge un fichier CSV et produit 1 Document LangChain par ligne.

    Stratégie de conversion :
      - Le contenu textuel du Document est une représentation lisible par un LLM :
            "colonne1: valeur1 | colonne2: valeur2 | ..."
      - Toutes les colonnes sont aussi stockées dans les métadonnées du Document
        pour permettre des filtres lors de la recherche (ex: filtrer par section).
      - text_columns permet de limiter le contenu textuel aux colonnes importantes
        pour la recherche sémantique (les autres restent en métadonnées seulement).

    Args:
        filepath:     Chemin vers le fichier CSV (UTF-8 attendu).
        text_columns: Colonnes à inclure dans le corps textuel du Document.
                      Si None → toutes les colonnes sont utilisées.

    Returns:
        Liste de Documents, une par ligne non-vide du CSV.
    """
    documents = []

    with filepath.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            logger.warning("CSV vide ou sans header : %s", filepath)
            return []

        for row_index, row in enumerate(reader):
            columns_for_text = text_columns or list(row.keys())

            text_parts = [
                f"{key}: {value.strip()}"
                for key, value in row.items()
                if key in columns_for_text and value and value.strip()
            ]

            if not text_parts:
                continue

            content = " | ".join(text_parts)

            metadata = {
                "filepath": str(filepath),
                "filename": filepath.stem,
                "section": filepath.parts[-2] if len(filepath.parts) > 1 else "root",
                "source_type": "csv",
                "row_index": row_index,
                **{k: v for k, v in row.items() if v and v.strip()},
            }

            documents.append(Document(page_content=content, metadata=metadata))

    logger.info("  ✓ %d lignes chargées depuis %s", len(documents), filepath.name)
    return documents


def load_documents(
    data_dir: str,
    max_file_size_mb: float = 10.0,
    target_files: list[Path] | None = None,
) -> list[Document]:
    """
    Point d'entrée du chargement : parcourt récursivement data_dir
    et charge tous les fichiers .md et .csv trouvés.

    Sécurités :
      - Limite de taille (max_file_size_mb) pour éviter de charger des fichiers énormes
      - Gestion des erreurs d'encodage et d'accès fichier
      - Les fichiers vides sont ignorés

    Métadonnées communes ajoutées à chaque Document Markdown :
      - filepath    : chemin absolu du fichier
      - filename    : nom sans extension
      - source_type : "markdown" ou "csv"
      - section     : sous-dossier parent (catégorie thématique)
      + les clés du frontmatter YAML si présent

    Args:
        data_dir:         Répertoire racine contenant les fichiers .md/.csv.
        max_file_size_mb: Taille maximale d'un fichier source (Mo).
        target_files:     Si fourni, seuls ces fichiers sont chargés (mode incrémental).
                          Si None, tous les fichiers de data_dir sont chargés (mode complet).

    Returns:
        Liste plate de Documents LangChain (tous types confondus).
    """
    documents = []
    data_path = Path(data_dir)
    max_bytes = max_file_size_mb * 1024 * 1024

    # Mode incrémental : on ne traite que les fichiers fournis explicitement
    # Mode complet    : on parcourt tout le dossier
    if target_files is not None:
        filepaths_md  = [f for f in target_files if f.suffix == ".md"]
        filepaths_csv = [f for f in target_files if f.suffix == ".csv"]
    else:
        filepaths_md  = sorted(data_path.rglob("*.md"))
        filepaths_csv = sorted(data_path.rglob("*.csv"))

    # — Chargement des fichiers Markdown (.md) —
    for filepath in filepaths_md:
        if filepath.stat().st_size > max_bytes:
            logger.warning("Fichier ignoré (trop grand, > %.0f Mo) : %s", max_file_size_mb, filepath)
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning("Fichier ignoré (%s) : %s", e, filepath)
            continue

        if not text.strip():
            continue

        metadata, content = parse_frontmatter(text)
        metadata["filepath"] = str(filepath)
        metadata["filename"] = filepath.stem
        metadata["source_type"] = "markdown"

        if "section" not in metadata:
            relative = filepath.relative_to(data_path)
            metadata["section"] = relative.parts[-2] if len(relative.parts) > 1 else "root"

        documents.append(Document(page_content=content, metadata=metadata))

    # — Chargement des fichiers CSV —
    for filepath in filepaths_csv:
        if filepath.stat().st_size > max_bytes:
            logger.warning("CSV ignoré (trop grand, > %.0f Mo) : %s", max_file_size_mb, filepath)
            continue

        try:
            csv_docs = load_csv(filepath)
            documents.extend(csv_docs)
        except (OSError, csv.Error) as e:
            logger.warning("CSV ignoré (%s) : %s", e, filepath)
            continue

    logger.info("  ✓ %d documents chargés depuis %s", len(documents), data_dir)
    return documents


# ---------------------------------------------------------------------------
# Splitter — découpage des documents en chunks pour l'embedding
# ---------------------------------------------------------------------------

def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 50,
) -> list[Document]:
    """
    Découpe les documents en « chunks » (morceaux) adaptés à l'embedding.

    Pourquoi découper ?
    → Les LLMs ont une fenêtre de contexte limitée.
    → L'embedding fonctionne mieux sur des passages courts et cohérents.
    → La recherche sémantique est plus précise sur des chunks ciblés.

    Stratégie en deux passes pour les fichiers Markdown :
      Passe 1 — MarkdownHeaderTextSplitter :
        Découpe sur les titres (#, ##, ###) pour respecter la structure
        logique du document. Enrichit les métadonnées de chaque chunk
        avec les titres parents (ex: {"h1": "Formation", "h2": "Calendrier"}).

      Passe 2 — RecursiveCharacterTextSplitter :
        Re-découpe les chunks encore trop longs (> chunk_size chars)
        en respectant l'ordre : paragraphes (\n\n) > lignes (\n) > mots > chars.
        chunk_overlap = zone de chevauchement entre chunks consécutifs
        pour éviter de couper le contexte à la frontière.

    Pour les CSV :
      Chaque ligne est déjà un Document court → on saute la passe Markdown
      et on applique directement RecursiveCharacterTextSplitter si nécessaire.

    Filtrage anti-bruit :
      Les chunks de moins de min_chunk_size caractères sont ignorés
      (titres seuls, lignes vides, artefacts de conversion).

    Args:
        documents:      Liste de Documents issus de load_documents().
        chunk_size:     Taille maximale d'un chunk en caractères.
        chunk_overlap:  Nombre de caractères partagés entre deux chunks adjacents.
        min_chunk_size: Seuil minimum pour conserver un chunk.

    Returns:
        Liste de Documents enrichis, prêts pour l'embedding.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=True,
    )

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        source_type = doc.metadata.get("source_type", "markdown")

        # — Traitement CSV : pas de header splitting —
        if source_type == "csv":
            content = doc.page_content.strip()
            if len(content) < min_chunk_size:
                continue
            if len(content) > chunk_size:
                sub_chunks = char_splitter.create_documents(
                    texts=[content],
                    metadatas=[doc.metadata],
                )
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(doc)
            continue

        # — Traitement Markdown : double passe —
        header_chunks = header_splitter.split_text(doc.page_content)

        for position, chunk in enumerate(header_chunks):
            content = chunk.page_content.strip()

            if len(content) < min_chunk_size:
                continue

            merged_metadata = {
                **doc.metadata,
                **chunk.metadata,
                "chunk_position": position,
                "total_chunks": len(header_chunks),
            }

            if len(content) > chunk_size:
                sub_chunks = char_splitter.create_documents(
                    texts=[content],
                    metadatas=[merged_metadata],
                )
                for i, sub in enumerate(sub_chunks):
                    sub.metadata["sub_chunk_index"] = i
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(Document(page_content=content, metadata=merged_metadata))

    logger.info("  ✓ %d chunks générés depuis %d documents", len(all_chunks), len(documents))
    return all_chunks


# ---------------------------------------------------------------------------
# Vectorstore — embedding et stockage ChromaDB
# ---------------------------------------------------------------------------

def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_collection",
    embedding_model_name: str = "BAAI/bge-m3",
    batch_size: int = 500,
) -> tuple[Chroma, dict[str, list[str]]]:
    """
    Calcule les embeddings de chaque chunk et les stocke dans ChromaDB.

    Qu'est-ce qu'un embedding ?
    → Un vecteur de nombres flottants (~1024 dimensions) qui représente
      le sens sémantique d'un texte. Deux textes similaires ont des vecteurs
      proches dans l'espace vectoriel.

    Modèle utilisé : BAAI/bge-m3
    → Modèle multilingue (français inclus), performant pour la recherche RAG.
    → Fonctionne en local, sans API externe.

    Stratégie d'upsert par batch :
    → On n'utilise pas Chroma.from_documents() car il ne gère pas les doublons.
    → On passe par le client ChromaDB natif et collection.upsert() :
         - Si l'ID existe déjà → mise à jour
         - Sinon → insertion
    → Les batches évitent les surcharges mémoire sur les gros corpus.

    Args:
        chunks:               Chunks issus de split_documents().
        persist_directory:    Dossier de persistance sur disque.
        collection_name:      Nom de la collection ChromaDB.
        embedding_model_name: Identifiant HuggingFace ou chemin local du modèle.
        batch_size:           Nombre de chunks traités simultanément.

    Returns:
        Tuple (instance Chroma, mapping filepath → liste de chunk_ids).

    Raises:
        ValueError: Si la liste de chunks est vide.
    """
    if not chunks:
        raise ValueError("La liste de chunks est vide, rien à ingérer.")

    device = _detect_device()
    logger.info("  → Chargement du modèle d'embedding : %s (device=%s)", embedding_model_name, device)

    embedding_fn = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )

    ids = _generate_chunk_ids(chunks)
    total = len(chunks)
    logger.info("  → Upsert de %d chunks par batches de %d...", total, batch_size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_ids = ids[start:end]
        batch_texts = [chunk.page_content for chunk in batch_chunks]

        batch_embeddings = embedding_fn.embed_documents(batch_texts)
        batch_metadatas = [_sanitize_metadata(chunk.metadata) for chunk in batch_chunks]

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

        logger.info("     ✓ Batch %d–%d upserted (%d/%d)", start, end, end, total)

    logger.info("  ✓ Vectorstore prêt — %d chunks dans '%s' (%s)", total, collection_name, persist_directory)

    # Construction du mapping filepath → liste de chunk_ids (pour le registre)
    filepath_to_ids: dict[str, list[str]] = {}
    for chunk, chunk_id in zip(chunks, ids):
        fp = chunk.metadata.get("filepath", "unknown")
        filepath_to_ids.setdefault(fp, []).append(chunk_id)

    return vectorstore, filepath_to_ids


# ---------------------------------------------------------------------------
# Collection management — gestion du cycle de vie de la collection
# ---------------------------------------------------------------------------

def _handle_existing_collection(
    persist_directory: str,
    collection_name: str,
    force_reingest: bool,
) -> None:
    """
    Vérifie l'état d'une collection ChromaDB existante avant l'ingestion.

    Deux comportements selon force_reingest :
      - True  → suppression complète + recréation (re-ingestion totale)
      - False → conservation + mise à jour par upsert (incrémental)

    Si le dossier persist_directory n'existe pas → rien à faire.
    """
    chroma_path = Path(persist_directory)
    if not chroma_path.exists():
        return

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        existing = [c.name for c in client.list_collections()]

        if collection_name not in existing:
            return

        count = client.get_collection(collection_name).count()

        if force_reingest:
            logger.info("[0/4] Collection '%s' existante (%d chunks) → suppression.", collection_name, count)
            client.delete_collection(collection_name)
        else:
            logger.info("  Collection '%s' existante (%d chunks) → mise à jour (upsert).", collection_name, count)

    except Exception as e:
        logger.warning("Impossible de lire la collection existante (%s) — elle sera recréée.", e)


# ---------------------------------------------------------------------------
# Summary — résumé de fin d'ingestion
# ---------------------------------------------------------------------------

def _print_summary(
    data_dir: str,
    n_docs: int,
    n_chunks: int,
    persist_directory: str,
    collection_name: str,
) -> None:
    """Affiche un récapitulatif lisible à la fin du pipeline."""
    logger.info("=" * 55)
    logger.info("   INGESTION TERMINÉE")
    logger.info("=" * 55)
    logger.info("  Source          : %s", data_dir)
    logger.info("  Documents       : %d", n_docs)
    logger.info("  Chunks          : %d", n_chunks)
    logger.info("  Collection      : %s", collection_name)
    logger.info("  Persistance     : %s", persist_directory)
    logger.info("=" * 55)


# ---------------------------------------------------------------------------
# Orchestrateur principal — pipeline complet
# ---------------------------------------------------------------------------

def run_ingestion(
    data_dir: str = str(BASE_DIR.parent / "data"),
    persist_directory: str = str(BASE_DIR.parent / "chroma_db"),
    collection_name: str = "rag_collection",
    embedding_model_name: str = str(BASE_DIR.parent / "models" / "bge-m3"),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 500,
    force_reingest: bool = False,
    max_file_size_mb: float = 10.0,
    registry_path: str | None = None,
) -> Chroma:
    """
    Orchestrateur du pipeline RAG complet : du fichier source au vectorstore.

    Mode incrémental (défaut, force_reingest=False) :
      - Compare chaque fichier .md/.csv au registre JSON (hash SHA-256)
      - N'ingère que les fichiers nouveaux ou modifiés
      - Supprime de ChromaDB les chunks des fichiers retirés du corpus
      - Met à jour le registre après chaque run
      → Idéal pour les runs automatiques : rapide si peu de changements.

    Mode réingestion complète (force_reingest=True) :
      - Supprime la collection ChromaDB et le registre existants
      - Repart de zéro sur tous les fichiers
      → À utiliser quand le modèle d'embedding change, ou pour forcer un recalcul total.

    Étapes :
      0. Chargement du registre + calcul du delta (nouveaux / modifiés / supprimés)
      1. Chargement des fichiers .md et .csv à traiter   [load_documents]
      2. Découpage en chunks                              [split_documents]
      3. Purge des chunks orphelins + Embedding + stockage [build_vectorstore]
      4. Mise à jour du registre

    Args:
        data_dir:             Répertoire racine contenant les fichiers .md/.csv.
        persist_directory:    Dossier de persistance ChromaDB sur disque.
        collection_name:      Nom de la collection Chroma.
        embedding_model_name: Chemin local du modèle bge-m3 pré-téléchargé.
        chunk_size:           Taille maximale d'un chunk en caractères.
        chunk_overlap:        Chevauchement entre chunks adjacents.
        batch_size:           Nombre de chunks par batch d'embedding.
        force_reingest:       True = tout supprimer et recommencer depuis zéro.
        max_file_size_mb:     Taille maximale d'un fichier source (Mo).
        registry_path:        Chemin du fichier registre JSON.
                              Par défaut : <persist_directory>/ingest_registry.json

    Returns:
        Instance Chroma prête à être utilisée comme retriever.

    Raises:
        FileNotFoundError: Si data_dir n'existe pas.
        ValueError:        Si aucun document .md ou .csv n'est trouvé.
    """
    from datetime import datetime, timezone

    logger.info("=" * 55)
    logger.info("   PIPELINE D'INGESTION RAG")
    logger.info("=" * 55)

    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Répertoire introuvable : {data_dir}")

    reg_path = Path(registry_path) if registry_path else Path(persist_directory) / "ingest_registry.json"

    # -----------------------------------------------------------------------
    # Étape 0a : réinitialisation complète si demandée
    # -----------------------------------------------------------------------
    if force_reingest:
        _handle_existing_collection(
            persist_directory=persist_directory,
            collection_name=collection_name,
            force_reingest=True,
        )
        if reg_path.exists():
            reg_path.unlink()
            logger.info("[0/4] Registre supprimé (réingestion complète).")

    # -----------------------------------------------------------------------
    # Étape 0b : chargement du registre + calcul du delta
    # -----------------------------------------------------------------------
    registry = _load_registry(reg_path)

    logger.info("[0/4] Analyse du corpus (mode %s)...",
                "COMPLET" if force_reingest else "INCRÉMENTAL")

    new_or_modified, deleted_filepaths = _compute_file_delta(
        data_dir=data_dir,
        registry=registry,
        max_file_size_mb=max_file_size_mb,
    )

    n_unchanged = sum(
        1 for p in registry
        if p not in deleted_filepaths
        and p not in [str(f) for f in new_or_modified]
    )
    logger.info(
        "  → %d nouveau(x)/modifié(s)  |  %d supprimé(s)  |  %d inchangé(s)",
        len(new_or_modified), len(deleted_filepaths), n_unchanged,
    )

    if not new_or_modified and not deleted_filepaths:
        logger.info("  ✓ Corpus à jour — aucune ingestion nécessaire.")
        client = chromadb.PersistentClient(path=persist_directory)
        embedding_fn = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": _detect_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_fn,
        )

    # -----------------------------------------------------------------------
    # Étape 1 : chargement des fichiers à (ré)ingérer uniquement
    # -----------------------------------------------------------------------
    logger.info("[1/4] Chargement des documents à ingérer...")
    documents = load_documents(
        data_dir,
        max_file_size_mb=max_file_size_mb,
        target_files=new_or_modified if new_or_modified else None,
    )

    if not documents and not deleted_filepaths:
        raise ValueError(f"Aucun fichier .md ou .csv trouvé dans : {data_dir}")

    # -----------------------------------------------------------------------
    # Étape 2 : découpage en chunks
    # -----------------------------------------------------------------------
    chunks = []
    if documents:
        logger.info("[2/4] Découpage en chunks...")
        chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # -----------------------------------------------------------------------
    # Étape 3 : purge des chunks orphelins + upsert des nouveaux
    # -----------------------------------------------------------------------
    logger.info("[3/4] Mise à jour ChromaDB...")

    _handle_existing_collection(
        persist_directory=persist_directory,
        collection_name=collection_name,
        force_reingest=False,
    )

    files_to_purge = deleted_filepaths + [str(f) for f in new_or_modified]
    if files_to_purge:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        _purge_deleted_chunks(collection, registry, files_to_purge)

    filepath_to_ids: dict[str, list[str]] = {}
    vectorstore = None

    if chunks:
        vectorstore, filepath_to_ids = build_vectorstore(
            chunks=chunks,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            batch_size=batch_size,
        )
    else:
        client = chromadb.PersistentClient(path=persist_directory)
        embedding_fn = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": _detect_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_fn,
        )

    # -----------------------------------------------------------------------
    # Étape 4 : mise à jour du registre
    # -----------------------------------------------------------------------
    logger.info("[4/4] Mise à jour du registre...")
    now_iso = datetime.now(timezone.utc).isoformat()

    for path in deleted_filepaths:
        registry.pop(path, None)

    for filepath in new_or_modified:
        str_path = str(filepath)
        registry[str_path] = {
            "hash": _file_hash(filepath),
            "ingested_at": now_iso,
            "chunk_ids": filepath_to_ids.get(str_path, []),
        }

    _save_registry(registry, reg_path)

    _print_summary(
        data_dir=data_dir,
        n_docs=len(documents),
        n_chunks=len(chunks),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vectorstore


# ---------------------------------------------------------------------------
# Point d'entrée — exécution directe du script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
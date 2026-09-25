# -*- coding: utf-8 -*-
"""
interface.py — Chat RAG avec Chainlit (ChromaDB + Groq)
Lancement : chainlit run interface.py
"""

from openpyxl import load_workbook
import os
import re
import random
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

import chainlit as cl
from groq import Groq
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Fix asyncio Windows
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR      = Path(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH        = SCRIPT_DIR / "interface.env"
load_dotenv(dotenv_path=ENV_PATH, verbose=True)

CHROMA_DIR      = SCRIPT_DIR.parent / "chroma_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = str(SCRIPT_DIR.parent / "models" / "bge-m3")
GROQ_MODEL      = os.getenv("GROQ_MODEL")
GROQ_LLM_API    = os.getenv("GROQ_LLM_API")
USERS_FILE      = SCRIPT_DIR.parent / "utilisateurs" / "users_esia.xlsx"
Path(SCRIPT_DIR / ".files").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Choix du modèle : on prend celui défini dans le .env OU un modèle QWEN
# ---------------------------------------------------------------------------

client = Groq(api_key=GROQ_LLM_API)
models = client.models.list()
liste_modele = []
for m in models.data:
    liste_modele.append(m.id)

if GROQ_MODEL not in liste_modele:  
    qwen_models = [m for m in liste_modele if "qwen" in m.lower()]
    GROQ_MODEL = qwen_models[0]
    
# ---------------------------------------------------------------------------
# Affichage des informations relatives à l'interface
# ---------------------------------------------------------------------------

print("[INFORMATION] Voici les chemins absolus de l'environnement")
print("              Si vous constatez une erreur, vérifiez le code.")
print("-" * 100)
print("=> Le fichier des utilisateurs est ici : ", USERS_FILE)
print("=> Le modèle d'embedding est situé ici : ", EMBEDDING_MODEL)
print("=> La base vectorielle est situé ici   : ", CHROMA_DIR)
print("=> Le modèle Groq utilisé est          : ", GROQ_MODEL)
print("=> Le répertoire du script est         : ", SCRIPT_DIR)
print("=> La clé API est                      :", "*" * 50)
print("-" * 100)


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------
with open("system_prompt.txt", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ---------------------------------------------------------------------------
# Utilisateurs
# ---------------------------------------------------------------------------
def load_users() -> dict:
    if not USERS_FILE.exists():
        print("[ERREUR] Le fichier des utilisateurs n'a pas été trouvé.")
        print("         Vérifiez le dossier 'utilisateurs', ici doit être présent le fichier : <users_esia.xlsx>")
        return {}
    try:
        wb = load_workbook(str(USERS_FILE), data_only=True)
        ws = wb["Utilisateurs"]
        users = {}
        for row in ws.iter_rows(min_row=5, values_only=True):
            if not row[3] or not row[4] or not row[5]:
                continue
            username = str(row[3]).strip().lower()
            password = str(row[4]).strip()
            role     = str(row[5]).strip().lower()
            if username and password and role in ("membre", "visiteur"):
                users[username] = {"password": password, "role": role}
        print(f"[INFORMATION] {len(users)} utilisateurs ont été extraits du fichier Excel !")
        print("-" * 100)
        print("Voici la liste : ")
        for key, values in users.items():
            print("    -> ", key, "( Rôle : ", values["role"], ")")
        print("-" * 100)
        return users
    except Exception as e:
        print(f"[ERREUR] Aucun utilisateur n'a été extrait du fichier utilisateur\nVérifiez le document et le format de rédaction.")
        print(f"-> : {e}")
        return {}

USERS = load_users()

SECTIONS_VISITEUR = ["Formation", "Candidature", "Insertion", "Ressources", "Contact", "Equipe", "Accueil"]
SECTIONS_MEMBRE   = SECTIONS_VISITEUR + ["Alumni"]

# ---------------------------------------------------------------------------
# Extraction des sources utilisées par le LLM
# ---------------------------------------------------------------------------
def extract_used_sources(answer: str, docs: list) -> tuple:
    match = re.search(r"SOURCES_USED:\s*(.+)$", answer, re.MULTILINE)
    if not match:
        return answer, docs
    used_filenames = [f.strip() for f in match.group(1).split(",")]
    used_docs = [d for d in docs if d.metadata.get("filename") in used_filenames]
    if not used_docs:
        used_docs = docs
    clean_answer = re.sub(r"\nSOURCES_USED:.+$", "", answer, flags=re.MULTILINE).strip()
    return clean_answer, used_docs

# ---------------------------------------------------------------------------
# Chargement ChromaDB + Embedding (une seule fois au démarrage)
# ---------------------------------------------------------------------------
print("[INFORMATION] Chargement du modèle d'embedding...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = "cpu"
print(f"[INFORMATION] L'appareil de calcul du PC pour l'embedding sera : {device.upper()}\n")

embedding_fn = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
print("[INFORMATION] Le modèle d'embedding (BGE-M3) est prêt à vectoriser les questions des utilisateurs !!")

if not CHROMA_DIR.exists():
    print("[ERREUR] Impossible de retrouver la base vectorielle")
    print("         Vérifiez que le dossier <chroma_db> est bien à la racine du dossier du projet")
else:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)
print("[INFORMATION] La base vectorielle a été trouvée et importée sur l'interface !!")

groq_client = Groq(api_key=GROQ_LLM_API.strip())
print("[INFORMATION] La clé API a bien été récupérée et sera prête à l'utilisation !!")

# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    user = USERS.get(username.lower().strip())
    if user and password == user["password"]:
        return cl.User(
            identifier=username,
            metadata={"role": user["role"]},
        )
    return None

# ---------------------------------------------------------------------------
# Starters — questions prédéfinies affichées sur l'écran d'accueil
# ---------------------------------------------------------------------------
STARTER_REPONSE = {
    "Contact Master ESA": (
        "**📬 Une question pour l'équipe pédagogique ?**\n\n"
        "➜ Contactez le Master ESA par e-mail à **master-esa@univ-orleans.fr**\n\n"
        "⚠️ **Important** : les messages rédigés dans cette interface ne sont pas transmis à l'équipe pédagogique. "
        "Pensez à les contacter directement pour garantir la prise en compte de votre demande."
    ),
    "Qui a créé ESActu ? 🤨": (
        "Interface développée par **Fatoumata, Marius et Jarod**, étudiants en Master ESA (Économétrie et Statistique Appliquée) "
        "à l'Université d'Orléans, dans le cadre d'un projet académique."
    ),
}

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="💼 Débouchés",
            message="Quels sont les débouchés du Master ESA ? Y'a t-il un secteur récurrent ?",
        ),
        cl.Starter(
            label="🎓 Formation",
            message="Quel est le contenu de la formation ? Commment sont articulés les maquettes de chaque année ?",
        ),
        cl.Starter(
            label="📊 Insertion",
            message="Quels sont les taux d'insertion ?",
        ),
        cl.Starter(
            label="🔄 Alternance",
            message="Le Master ESA propose-t-il de l'alternance ?",
        ),
        cl.Starter(
            label="👤 Profils admis",
            message="Quels profils sont recherchés pour candidater ? C'est quoi le taux d'admission selon le profil de l'étudiant ?",
        ),
    ]

# ---------------------------------------------------------------------------
# Animation thinking
# ---------------------------------------------------------------------------
async def animer_thinking(msg: cl.Message, texte_base: str, stop_event: asyncio.Event):
    points = ["", ".", "..", "..."]
    i = 0
    while not stop_event.is_set():
        msg.content = texte_base + points[i % 4]
        await msg.update()
        await asyncio.sleep(0.4)
        i += 1


# ---------------------------------------------------------------------------
# Réception d'un message — pipeline RAG
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    THINKING_MESSAGES = [
        "🎓 ESActu parcourt la documentation du Master ESA",
        "📄 ESActu est en train de lire les documents sur le sujet",
        "🔍 ESActu fouille dans la base documentaire",
        "📚 ESActu consulte les ressources du Master ESA",
        "🗂️ ESActu feuillette les dossiers de formation",
        "📖 ESActu lit attentivement les articles concernés",
        "🧐 ESActu lit un document qui semble pertinent",
        "📋 ESActu prépare une réponse à partir des sources",
        "🔎 ESActu recherche les informations pertinentes",
        "📊 ESActu consulte les données d'insertion professionnelle",
        "📑 ESActu consulte les pages du site master-esa.fr",
    ]

    user = cl.user_session.get("user")
    role = user.metadata.get("role", "visiteur") if user else "visiteur"
    question = message.content.strip()

    # -- Réponses starters prédéfinies --
    if question in STARTER_REPONSE:
        await cl.Message(content=STARTER_REPONSE[question], author="ESActu").send()
        return

    if not question:
        return

    allowed = SECTIONS_MEMBRE if role == "membre" else SECTIONS_VISITEUR

    # -- Animation thinking --
    texte_base = random.choice(THINKING_MESSAGES)
    thinking_msg = cl.Message(content=texte_base, author=" ")
    await thinking_msg.send()
    stop_event = asyncio.Event()
    animation_task = asyncio.create_task(animer_thinking(thinking_msg, texte_base, stop_event))

    # -- Recherche ChromaDB --
    docs = await vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "lambda_mult": 0.7},
    ).ainvoke(question)
    docs = [d for d in docs if d.metadata.get("section", "Accueil") in allowed]

    # -- Arrêt de l'animation --
    stop_event.set()
    await animation_task

    if not docs:
        await cl.Message(
            content="❌ Aucun document pertinent trouvé. Essayez de reformuler.",
            author="ESActu",
        ).send()
        return

    # -- Construction du contexte --
    context = "\n\n---\n\n".join(
        "\n".join([
            f"[Source {i+1}]",
            f"Titre    : {d.metadata.get('title', d.metadata.get('filename', ''))}",
            f"Section  : {d.metadata.get('section', '')}",
            f"URL      : {d.metadata.get('url', '')}",
            f"\n{d.page_content}",
        ])
        for i, d in enumerate(docs)
    )

    # -- Appel Groq avec streaming --
    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Documents :\n{context}\n\nQuestion : {question}"},
        ],
        temperature=0.1,
        max_tokens=1048,
        stream=True,
    )

    # -- Message streamé --
    response_msg = cl.Message(content="", author="ESActu")
    await response_msg.send()
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            await response_msg.stream_token(delta)

    # -- Extraction des questions annexes --
    full_response = response_msg.content
    questions_annexes = re.findall(
        r"\*\*(.+?)\*\*",
        re.search(r"###\s*❓\s*Questions annexes(.+?)(?=##|\Z)", full_response, re.DOTALL).group(1)
        if re.search(r"###\s*❓\s*Questions annexes", full_response) else ""
    )
    clean_response = re.sub(
        r"###\s*❓\s*Questions annexes.+?(?=##|\Z)",
        "",
        full_response,
        flags=re.DOTALL
    ).strip()


    # -- Mise à jour : réponse nettoyée + logo --
    response_msg.content = clean_response
    await response_msg.update()

    # -- Boutons questions annexes --
    if questions_annexes:
        actions = [
            cl.Action(
                name="question_annexe",
                label=f"❓ {q}",
                payload={"question": q},
            )
            for q in questions_annexes
        ]
        await cl.Message(
            content="**Vous pourriez aussi être intéressé par :**",
            actions=actions,
            author="ESActu",
        ).send()

@cl.on_chat_start
async def on_start():
    pass

# -- Gestion du clic sur une question annexe --
@cl.action_callback("question_annexe")
async def on_question_annexe(action: cl.Action):
    question = action.payload.get("question", "")
    await cl.Message(content=question, author="Vous").send()
    await on_message(cl.Message(content=question))
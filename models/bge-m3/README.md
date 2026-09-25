# Modèle BGE-M3

Ce dossier doit contenir le modèle d'embedding **BAAI/bge-m3**, utilisé par `ingest.py` et `interface.py` pour vectoriser les documents et les questions.

Le modèle est disponible sur HuggingFace : [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

Les fichiers volumineux (`model.safetensors`, `tokenizer.json`, etc.) sont exclus du dépôt Git — il faut les télécharger manuellement avant de lancer le projet.

## Option 1 — Via `sentence-transformers` (recommandé)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.save("./models/bge-m3")
```

## Option 2 — Via `huggingface-cli`

```bash
pip install huggingface_hub
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
```

## Option 3 — Sans téléchargement local (déconseillé pour la prod)

Remplacer directement le chemin local par l'identifiant HuggingFace dans `ingest.py` et `interface.py` :

```python
EMBEDDING_MODEL = "BAAI/bge-m3"
```

Le modèle sera alors téléchargé automatiquement au premier lancement et mis en cache dans `~/.cache/huggingface/`. Moins pratique pour un déploiement figé (Render, Docker), car le téléchargement (~2 Go) se refait à chaque build si le cache n'est pas persistant.

## Fichiers attendus après téléchargement local

```
models/bge-m3/
├── config.json
├── config_sentence_transformers.json
├── model.safetensors          # ~2.2 Go
├── modules.json
├── README.md
├── sentence_bert_config.json
├── tokenizer.json
├── tokenizer_config.json
├── 1_Pooling/
│   └── config.json
└── 2_Normalize/
```

## Vérification

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/bge-m3")
vecteur = model.encode("Test d'embedding")
print(len(vecteur))  # doit afficher 1024
```

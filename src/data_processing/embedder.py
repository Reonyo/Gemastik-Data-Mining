"""
Embeddings & ChromaDB Indexer

- Loads BAAI/bge-m3 (SentenceTransformers)
- Reads data/processed/legal_knowledge_base.jsonl
- Computes embeddings in batches
- Upserts into a persistent ChromaDB collection with metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EmbedConfig:
	model_name: str = "BAAI/bge-m3"
	input_jsonl: Path = Path("data/processed/legal_knowledge_base.jsonl")
	persist_dir: Path = Path("chroma_db")
	collection_name: str = "legal_kb"
	batch_size: int = 128


def load_records(path: Path) -> List[Dict]:
	if not path.exists():
		raise FileNotFoundError(f"Input JSONL not found: {path}")
	recs = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			recs.append(json.loads(line))
	return recs


def build_text_and_meta(rec: Dict) -> Tuple[str, Dict]:
	text = rec.get("content") or rec.get("text") or ""
	meta = rec.get("metadata", {})
	return text, meta


def index_into_chroma(cfg: EmbedConfig) -> None:
	# Init embedding model
	logger.info(f"Loading embedding model: {cfg.model_name}")
	model = SentenceTransformer(cfg.model_name)

	# Init Chroma client and collection
	logger.info(f"Initializing Chroma at: {cfg.persist_dir}")
	client = chromadb.Client(Settings(persist_directory=str(cfg.persist_dir), is_persistent=True))
	collection = client.get_or_create_collection(cfg.collection_name, metadata={"hnsw:space": "cosine"})

	# Load records
	logger.info(f"Reading records: {cfg.input_jsonl}")
	records = load_records(cfg.input_jsonl)
	logger.info(f"Loaded {len(records)} records")

	# Prepare batches
	ids: List[str] = []
	docs: List[str] = []
	metas: List[Dict] = []

	for i, rec in enumerate(records):
		text, meta = build_text_and_meta(rec)
		if not text:
			continue
		rid = f"kb_{i:06d}"
		ids.append(rid)
		docs.append(text)
		metas.append(meta)

		# Flush in batches
		if len(docs) >= cfg.batch_size:
			embs = model.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()
			collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
			ids, docs, metas = [], [], []

	# Flush remainder
	if docs:
		embs = model.encode(docs, show_progress_bar=False, normalize_embeddings=True).tolist()
		collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

	# Persist index
	client.persist()
	logger.info(f"Indexing complete. Collection '{cfg.collection_name}' persisted at {cfg.persist_dir}")


def query_demo(cfg: EmbedConfig, query: str, top_k: int = 5) -> List[Dict]:
	client = chromadb.Client(Settings(persist_directory=str(cfg.persist_dir), is_persistent=True))
	collection = client.get_or_create_collection(cfg.collection_name)
	# Use same model for query encoding
	model = SentenceTransformer(cfg.model_name)
	q_emb = model.encode([query], normalize_embeddings=True).tolist()
	res = collection.query(query_embeddings=q_emb, n_results=top_k, include=["documents", "metadatas", "distances", "ids"])
	results = []
	for i in range(len(res["ids"][0])):
		results.append({
			"id": res["ids"][0][i],
			"distance": res.get("distances", [[None]])[0][i],
			"document": res["documents"][0][i],
			"metadata": res["metadatas"][0][i],
		})
	return results


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Embed and index legal knowledge base into ChromaDB")
	parser.add_argument("--mode", choices=["index", "query"], default="index")
	parser.add_argument("--query", default="apa itu penjelasan pasal?")
	parser.add_argument("--top-k", type=int, default=5)
	args = parser.parse_args()

	cfg = EmbedConfig()

	if args.mode == "index":
		index_into_chroma(cfg)
	else:
		hits = query_demo(cfg, args.query, args.top_k)
		for i, h in enumerate(hits, 1):
			print(f"{i}. id={h['id']} dist={h['distance']}")
			print(f"   meta={h['metadata']}")
			print(f"   doc = {h['document'][:160]}...")


if __name__ == "__main__":
	main()

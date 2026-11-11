"""Vector search over document chunks and past episodes, with an HNSW index.

HNSW (usearch, in-process): approximate nearest neighbour in O(log n). Parameters:
  M (connectivity)=16         graph degree; recall vs. memory
  ef_construction=128         build-time beam width; recall vs. build time
  ef_search=64                query-time beam width; recall vs. latency
The index is *derived*: vectors are stored in the DB next to their rows
(doc_chunks.embedding, episode_vectors.embedding) and the index is rebuilt from
them on start. The DB stays the source of truth, and the index can be rebuilt
or re-tuned at any time.

Metadata filtering. Every vector carries a payload (hire_id, doc_type,
classification, doc_hash). `search(where=...)` runs the filter during
retrieval, not after: it widens the HNSW candidate set (k, 4k, 16k, ...) until
k results pass the filter; if HNSW still can't find k (it is approximate and may
not reach every node), it falls back to an exact scan over the filtered subset.
A selective filter (a user who can read 2 documents out of 10,000) therefore
returns everything it should, and nothing the caller can't read is returned. Production: pgvector's
HNSW index with the ACL as a SQL WHERE clause and iterative index scans, so
the vectors, the filter and the permissions sit in one transactional store.

Embeddings. HashingEmbedder is a deterministic, offline feature-hashing
embedder (word unigrams, bigrams, character 4-grams): lexical rather than
semantic, but it needs no new vendor. The `embed_model` column records which
embedder produced each vector, so switching to a semantic model (e.g. Vertex
AI text embeddings, if that is approved for the stack) triggers a re-embed
instead of mixing vector spaces.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional

import numpy as np
from usearch.index import Index

from .chunking import chunk_text
from .store import Store

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 4]}" for i in range(max(1, len(padded) - 3))]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 128, ef_search: int = 64):
        self.index = Index(ndim=dim, metric="cos", dtype="f32", connectivity=M,
                           expansion_add=ef_construction, expansion_search=ef_search)
        self._key: dict[str, int] = {}
        self._id: dict[int, str] = {}
        self.payload: dict[str, dict] = {}
        self._vec: dict[str, np.ndarray] = {}  # kept for the exact fallback
        self.exact_fallbacks = 0

    def __len__(self) -> int:
        return len(self._key)

    def add(self, item_id: str, vector: np.ndarray, payload: dict) -> bool:
        if item_id in self._key:
            return False  # idempotent: the same chunk/episode is indexed once
        key = len(self._key) + 1
        self._key[item_id], self._id[key] = key, item_id
        self.payload[item_id] = payload
        self._vec[item_id] = vector.astype(np.float32)
        self.index.add(key, self._vec[item_id])
        return True

    def search(self, vector: np.ndarray, k: int,
               where: Optional[Callable[[dict], bool]] = None) -> list[tuple[str, float, dict]]:
        if not self._key:
            return []
        fetch = k
        while True:
            fetch = min(fetch, len(self._key))
            matches = self.index.search(vector.astype(np.float32), fetch)
            hits = []
            for key, dist in zip(np.atleast_1d(matches.keys), np.atleast_1d(matches.distances)):
                item = self._id.get(int(key))
                if item is None:
                    continue
                payload = self.payload[item]
                if where is None or where(payload):
                    hits.append((item, round(1.0 - float(dist), 4), payload))
            if len(hits) >= k:
                return hits[:k]
            if fetch >= len(self._key):
                break
            fetch *= 4  # filter is selective: widen the candidate set
        # HNSW is approximate: even asked for every neighbour it may not reach every
        # node (here: 245 of 300). With a selective filter that silently drops
        # allowed results, so fall back to an exact scan over the items that pass
        # the filter - small by definition when we get here.
        self.exact_fallbacks += 1
        allowed = [i for i, p in self.payload.items() if where is None or where(p)]
        if not allowed:
            return []
        mat = np.stack([self._vec[i] for i in allowed])
        q = vector.astype(np.float32)
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-12)
        order = np.argsort(-sims)[:k]
        return [(allowed[j], round(float(sims[j]), 4), self.payload[allowed[j]]) for j in order]


class VectorStore:
    def __init__(self, store: Store, embedder: Optional[HashingEmbedder] = None):
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.chunks = HNSWIndex(self.embedder.dim)
        self.episodes = HNSWIndex(self.embedder.dim)

    def _vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def rebuild(self) -> None:
        """Rebuild both indexes from the DB; re-embed rows from an older embedder."""
        meta: dict[str, dict] = {}
        for c in self.store.all_chunks():
            if c["doc_hash"] not in meta:  # ACL metadata lives on the document, not the chunk
                d = self.store.get_document(c["doc_hash"]) or {}
                meta[c["doc_hash"]] = {"doc_type": d.get("doc_type"), "classification": d.get("classification")}
            vec = (self._vec(c["embedding"]) if c["embed_model"] == self.embedder.name
                   else self.embedder.embed([f"{c['heading']}\n{c['text']}" if c["heading"] else c["text"]])[0])
            self.chunks.add(f"{c['doc_hash']}:{c['idx']}", vec, self._chunk_payload({**c, "doc": meta[c["doc_hash"]]}))
        for row in self.store.episode_vectors():
            ep = self.store.episode(row["episode_id"])
            if not ep:
                continue
            vec = (self._vec(row["embedding"]) if row["embed_model"] == self.embedder.name
                   else self.embedder.embed([self._episode_text(ep)])[0])
            self.episodes.add(ep["episode_id"], vec, {"run_id": ep["run_id"], "hire_id": ep["hire_id"]})

    # ------------------------------------------------------------ documents
    @staticmethod
    def _chunk_payload(c: dict) -> dict:
        doc = c.get("doc") or {}
        return {"doc_hash": c["doc_hash"], "idx": c["idx"], "hire_id": c["hire_id"],
                "char_start": c["char_start"], "char_end": c["char_end"], "kind": c.get("kind"),
                "heading": c.get("heading"),
                "doc_type": doc.get("doc_type"), "classification": doc.get("classification")}

    def index_document(self, doc_hash: str, hire_id: str, text: str, doc_type: str, classification: str) -> int:
        chunks = chunk_text(text, embed=self.embedder.embed)  # adaptive: structure + semantic cuts
        if not chunks:
            return 0
        vecs = self.embedder.embed([c.embed_text() for c in chunks])  # heading-contextualised
        self.store.insert_chunks([(doc_hash, c.idx, hire_id, c.start, c.end, c.text, self.embedder.name,
                                   vecs[i].tobytes(), c.kind, c.heading) for i, c in enumerate(chunks)])
        meta = {"doc_type": doc_type, "classification": classification}
        added = 0
        for i, c in enumerate(chunks):
            added += self.chunks.add(f"{doc_hash}:{c.idx}", vecs[i], self._chunk_payload(
                {"doc_hash": doc_hash, "idx": c.idx, "hire_id": hire_id, "char_start": c.start,
                 "char_end": c.end, "kind": c.kind, "heading": c.heading, "doc": meta}))
        return added

    def search_chunks(self, query: str, k: int = 5, where: Optional[Callable[[dict], bool]] = None) -> list[dict]:
        hits = self.chunks.search(self.embedder.embed([query])[0], k, where)
        texts = {}
        out = []
        for item_id, sim, p in hits:
            if p["doc_hash"] not in texts:
                texts[p["doc_hash"]] = {c["idx"]: c["text"] for c in self.store.chunks_for(p["doc_hash"])}
            out.append({**p, "score": sim, "text": texts[p["doc_hash"]].get(p["idx"], "")})
        return out

    # ------------------------------------------------------------- episodes
    @staticmethod
    def _episode_text(ep: dict) -> str:
        return f"{ep['situation']} {' '.join(ep['features'])}"

    def index_episode(self, ep: dict) -> None:
        vec = self.embedder.embed([self._episode_text(ep)])[0]
        self.store.upsert_episode_vector(ep["episode_id"], self.embedder.name, vec.tobytes())
        self.episodes.add(ep["episode_id"], vec, {"run_id": ep["run_id"], "hire_id": ep["hire_id"]})

    def search_episodes(self, text: str, k: int = 50) -> dict[str, float]:
        return {item: sim for item, sim, _ in self.episodes.search(self.embedder.embed([text])[0], k)}

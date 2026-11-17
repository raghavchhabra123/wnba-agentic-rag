"""Chunking, embedding and search over play-by-play text.

Design choices (and why they differ from the course version):
  * A chunk is a window of consecutive plays from ONE game, not one isolated
    line. Each line carries quarter, clock, team and running score, and the
    chunk header names the game, so retrieved text can be cited and ordered.
  * Embeddings are built once and saved to data/index/ (the course version
    re-uploaded every play on every run, duplicating the index).
  * Search can be scoped to a single game_id. Unscoped search is kept only to
    reproduce the naive-RAG baseline.

Usage:
    python -m wnba_rag.retrieval build
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from wnba_rag.config import EMBED_MODEL, get_openai_client
from wnba_rag.paths import INDEX_DIR
from wnba_rag.stats import load_tables

WINDOW = 12   # plays per chunk
STRIDE = 10   # 2-play overlap so a sequence is not cut in half
SKIP_TYPES = {"Substitution"}  # line changes add noise without answering questions


def build_chunks() -> pd.DataFrame:
    t = load_tables()
    games = t.games.set_index("game_id")
    rows = []
    for game_id, plays in t.plays.groupby("game_id", sort=True):
        g = games.loc[game_id]
        plays = plays[~plays["type"].isin(SKIP_TYPES)].reset_index(drop=True)
        header_game = f"{g['away']} at {g['home']}, {g['date']} ({g['season_type']})"
        for start in range(0, len(plays), STRIDE):
            window = plays.iloc[start:start + WINDOW]
            if window.empty:
                break
            lines = []
            for p in window.itertuples():
                team = f"[{p.team}] " if isinstance(p.team, str) else ""
                lines.append(f"Q{p.period} {p.clock} {team}{p.text} "
                             f"(score {g['away_abbr']} {p.away_score}-{p.home_score} {g['home_abbr']})")
            first, last = window.iloc[0], window.iloc[-1]
            header = f"{header_game} | Q{first.period} {first.clock} to Q{last.period} {last.clock}"
            rows.append({
                "chunk_id": len(rows), "game_id": int(game_id),
                "period_start": int(first.period), "period_end": int(last.period),
                "text": header + "\n" + "\n".join(lines),
            })
            if start + WINDOW >= len(plays):
                break
    return pd.DataFrame(rows)


def _embed(texts: list[str], batch: int = 256) -> np.ndarray:
    client = get_openai_client()
    vecs = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        vecs.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    print(f"{len(chunks)} chunks from {chunks['game_id'].nunique()} games")
    vectors = _embed(chunks["text"].tolist())
    # Stored as float16 (33 MB instead of 65 MB, small enough to commit for deployment).
    # It keeps 99.2% of the same top-6 results as float32.
    np.save(INDEX_DIR / "vectors_f16.npy", vectors.astype("float16"))
    chunks.to_csv(INDEX_DIR / "chunks.csv", index=False)
    (INDEX_DIR / "meta.json").write_text(json.dumps(
        {"embed_model": EMBED_MODEL, "window": WINDOW, "stride": STRIDE, "n_chunks": len(chunks)}))
    print("Index saved to", INDEX_DIR)


class PlayIndex:
    def __init__(self) -> None:
        self.chunks = pd.read_csv(INDEX_DIR / "chunks.csv")
        self.vectors = np.load(INDEX_DIR / "vectors_f16.npy").astype("float32")  # FAISS needs float32
        self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self.faiss.add(self.vectors)
        self._by_game = self.chunks.groupby("game_id").indices

    def _query_vector(self, query: str) -> np.ndarray:
        return _embed_query(query)

    def search(self, query: str, k: int = 6, game_id: int | None = None,
               periods: list[int] | None = None) -> list[dict]:
        q = self._query_vector(query)
        if game_id is None:
            scores, idx = self.faiss.search(q[None, :], k)
            hits = list(zip(idx[0], scores[0]))
        else:
            ids = np.asarray(self._by_game.get(game_id, []))
            if periods:
                sub = self.chunks.iloc[ids]
                keep = sub["period_end"].ge(min(periods)) & sub["period_start"].le(max(periods))
                ids = ids[keep.to_numpy()]
            if len(ids) == 0:
                return []
            scores = self.vectors[ids] @ q
            order = np.argsort(-scores)[:k]
            hits = list(zip(ids[order], scores[order]))
        return [{"chunk_id": int(i), "game_id": int(self.chunks.iloc[i]["game_id"]),
                 "score": round(float(s), 4), "text": self.chunks.iloc[i]["text"]} for i, s in hits]


@lru_cache(maxsize=2048)
def _embed_query(query: str) -> np.ndarray:
    client = get_openai_client()
    v = np.asarray(client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding,
                   dtype="float32")
    return v / np.linalg.norm(v)


@lru_cache(maxsize=1)
def get_index() -> PlayIndex:
    return PlayIndex()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print(__doc__)

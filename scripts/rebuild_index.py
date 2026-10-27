"""Reparse originals and atomically publish a validated parent/child index."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-tokens", type=int, default=100)
    parser.add_argument("--overlap-tokens", type=int, default=16)
    parser.add_argument("--parent-tokens", type=int, default=500)
    parser.add_argument("--mode", choices=["smart", "fixed"], default="smart")
    parser.add_argument("--data-dir", type=Path)
    options = parser.parse_args()
    from src.chunk_config import ChunkSettings
    from src.index_builder import rebuild_library
    settings = ChunkSettings(child_target_tokens=options.child_tokens, overlap_tokens=options.overlap_tokens, parent_target_tokens=options.parent_tokens, mode=options.mode)
    snapshot = rebuild_library(settings=settings, data_dir=options.data_dir, progress=lambda message: print(message, flush=True))
    print(json.dumps({"revision": snapshot.revision, "validation": snapshot.manifest["validation"], "embeddings": snapshot.manifest["embeddings"]}, ensure_ascii=False, indent=2))

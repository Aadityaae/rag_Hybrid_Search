"""Seeds the pipeline: generates the synthetic corpus if data/raw is empty,
then ingests it into the vector + BM25 stores. Run this once before
starting the API/dashboard.

Usage: python scripts/seed.py [--strategy fixed|recursive|semantic]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DIR
from app.ingestion.pipeline import ingest_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="recursive", choices=["fixed", "recursive", "semantic"])
    args = parser.parse_args()

    if not any(RAW_DIR.iterdir()):
        print("data/raw is empty -- generating synthetic corpus...")
        from scripts.generate_synthetic_corpus import main as gen_corpus
        gen_corpus()

    print(f"Ingesting {RAW_DIR} with strategy='{args.strategy}'...")
    stats = ingest_directory(RAW_DIR, strategy=args.strategy)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

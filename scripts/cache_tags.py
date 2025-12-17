"""
Cache Tags Script.

Extracts tags from metadata CSVs, counts frequencies, and saves top N tags
to cache directory for fast loading during app initialization.
"""
import os
import sys
import csv
import pickle
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# Add project root to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
subproject_root = os.path.abspath(os.path.join(current_dir, ".."))
project_root = os.path.abspath(os.path.join(subproject_root, "../../../.."))

sys.path.insert(0, subproject_root)

def load_tags_from_csvs(metadata_dir: str, max_tags: int = 2000) -> Tuple[List[str], Dict[str, int]]:
    """
    Load tags from metadata CSVs and return top N tags with their frequencies.
    
    Args:
        metadata_dir: Directory containing id_tags.csv and id_genres.csv
        max_tags: Maximum number of tags to return (top N by frequency)
        
    Returns:
        Tuple of (tag_list, frequency_dict)
    """
    md = Path(metadata_dir)
    tags_path = md / "id_tags.csv"
    genres_path = md / "id_genres.csv"
    
    counter: Counter[str] = Counter()
    
    def _consume_csv(path: Path, field: str) -> None:
        """Read CSV and count tag frequencies."""
        if not path.exists():
            print(f"[cache_tags] Warning: {path} not found, skipping.")
            return
            
        print(f"[cache_tags] Reading {path}...")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            count = 0
            for row in reader:
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                for tok in raw.split(","):
                    t = tok.strip()
                    if t:
                        counter[t] += 1
                count += 1
                if count % 10000 == 0:
                    print(f"  Processed {count} rows...")
        print(f"  Done. Processed {count} rows total.")
    
    _consume_csv(tags_path, "tags")
    _consume_csv(genres_path, "genres")
    
    print(f"[cache_tags] Total unique tags: {len(counter)}")
    
    # Get top N tags
    most_common = counter.most_common(max_tags)
    tag_list = [t for t, _ in most_common]
    freq_dict = {t: f for t, f in most_common}
    
    print(f"[cache_tags] Extracted top {len(tag_list)} tags.")
    return tag_list, freq_dict

def save_tags_cache(
    tags: List[str],
    frequencies: Dict[str, int],
    cache_path: str,
    format: str = "pickle"
):
    """
    Save tags and frequencies to cache file.
    
    Args:
        tags: List of tags (ordered by frequency, descending)
        frequencies: Dictionary mapping tag -> frequency count
        cache_path: Output file path
        format: "pickle" or "json"
    """
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "tags": tags,
        "frequencies": frequencies,
        "total_tags": len(tags)
    }
    
    if format == "pickle":
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache_tags] Saved to {cache_file} (pickle format)")
    elif format == "json":
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[cache_tags] Saved to {cache_file} (JSON format)")
    else:
        raise ValueError(f"Unknown format: {format}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache tags from metadata CSVs"
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Path to metadata directory (default: auto-detect)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache directory (default: auto-detect)"
    )
    parser.add_argument(
        "--max_tags",
        type=int,
        default=2000,
        help="Maximum number of tags to cache (top N by frequency)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pickle", "json", "both"],
        default="pickle",
        help="Output format: pickle (fast), json (human-readable), or both"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="tags_cache",
        help="Output filename (without extension)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect paths if not provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subproject_dir = os.path.dirname(script_dir)
    
    if args.metadata_dir is None:
        args.metadata_dir = os.path.join(subproject_dir, "data", "metadata")
    if args.cache_dir is None:
        args.cache_dir = os.path.join(subproject_dir, "data", "cache")
    
    print(f"[cache_tags] Metadata directory: {args.metadata_dir}")
    print(f"[cache_tags] Cache directory: {args.cache_dir}")
    print(f"[cache_tags] Max tags: {args.max_tags}")
    
    # Load tags
    tags, frequencies = load_tags_from_csvs(args.metadata_dir, max_tags=args.max_tags)
    
    # Save cache
    if args.format in ["pickle", "both"]:
        cache_path = os.path.join(args.cache_dir, f"{args.output_name}.pkl")
        save_tags_cache(tags, frequencies, cache_path, format="pickle")
    
    if args.format in ["json", "both"]:
        cache_path = os.path.join(args.cache_dir, f"{args.output_name}.json")
        save_tags_cache(tags, frequencies, cache_path, format="json")
    
    # Print summary
    print(f"\n[cache_tags] Summary:")
    print(f"  Total tags cached: {len(tags)}")
    print(f"  Top 10 tags: {tags[:10]}")
    print(f"  Cache saved to: {args.cache_dir}")

if __name__ == "__main__":
    main()



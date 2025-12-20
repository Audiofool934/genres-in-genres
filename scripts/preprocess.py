
"""
Preprocess Script.

Scans the data/music directory, extracts features using MuLan, and caches the results.
Usage: python scripts/preprocess.py
"""
import os
import sys
import argparse

# Add project root to python path to allow absolute imports of 'src'
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../scripts
subproject_root = os.path.abspath(os.path.join(current_dir, "..")) # .../genres-in-genres

sys.path.insert(0, subproject_root) # For src.xxx

from src.library_manager import LibraryManager
from src.extractor import MuLanFeatureExtractor

def preprocess(data_dir, cache_dir, device="cuda", artist: str | None = None):
    print(f"Scanning library at: {data_dir}")
    print(f"Cache location: {cache_dir}")
    
    manager = LibraryManager(data_dir, cache_dir)
    extractor = MuLanFeatureExtractor(device=device)

    # 2. Find all artists in music/ dir
    artist_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if artist:
        artist_dirs = [a for a in artist_dirs if a == artist]
    
    if not artist_dirs:
        print("No artist directories found.")
        return

    for artist in artist_dirs:
        print(f"\nProcessing Artist: {artist}")
        career = manager.scan_artist(artist)
        if not career.tracks:
            print(f"No tracks found for {artist}. Skipping.")
            continue
        
        extractor.process_career(career)
        manager.save_to_cache(career)

    print("\nPre-processing complete.")

if __name__ == "__main__":
    # Determine default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subproject_dir = os.path.dirname(script_dir)
    default_data_dir = os.path.join(subproject_dir, "data", "music")
    default_cache_dir = os.path.join(subproject_dir, "data", "cache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=default_data_dir, help="Path to music library")
    parser.add_argument("--cache_dir", default=default_cache_dir, help="Path to output cache")
    parser.add_argument("--device", default="cuda", help="Device to run model on")
    parser.add_argument("--artist", default=None, help="Optional: process a single artist directory name")
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.cache_dir, args.device, artist=args.artist)

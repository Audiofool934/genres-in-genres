"""
Preprocess Script.

Scans the data/music directory, extracts audio features using MuQ-MuLan, and caches the results.
Usage: python scripts/preprocess.py --device cuda
"""
import os
import sys
import argparse
import pickle
import torch
import librosa
import numpy as np
from tqdm import tqdm

# Add project root to python path to allow absolute imports of 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts
subproject_root = os.path.abspath(os.path.join(current_dir, ".."))  # .../genres-in-genres
sys.path.insert(0, subproject_root)

from src.library_manager import LibraryManager
from src.core import TrackEmbedding


class MuQAudioExtractor:
    """
    Audio feature extractor using the official MuQ-MuLan library.
    Extracts 512-dimensional audio embeddings.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"[MuQAudioExtractor] Initializing on {device}...")
        
        try:
            from muq import MuQMuLan
            self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
            self.model = self.model.to(device).eval()
            print("[MuQAudioExtractor] Model loaded successfully.")
        except Exception as e:
            print(f"[MuQAudioExtractor] Failed to load model: {e}")
            raise e
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract audio embedding for a single file.
        Returns: np.ndarray of shape [512]
        """
        try:
            # Load audio at 24kHz (MuQ requirement)
            wav, sr = librosa.load(audio_path, sr=24000)
            wav_tensor = torch.tensor(wav).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(wavs=wav_tensor)
                # Normalize
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().squeeze()
        
        except Exception as e:
            print(f"[MuQAudioExtractor] Error processing {audio_path}: {e}")
            return None
    
    def process_career(self, career):
        """
        Process all tracks in a career and attach embeddings.
        """
        print(f"[MuQAudioExtractor] Processing {len(career.tracks)} tracks...")
        
        for track in tqdm(career.tracks, desc=f"Extracting {career.artist_name}"):
            embedding = self.extract_embedding(track.file_path)
            if embedding is not None:
                career.embeddings.append(
                    TrackEmbedding(track_ref=track, vector=embedding)
                )
        
        print(f"[MuQAudioExtractor] Extracted {len(career.embeddings)} embeddings.")


def preprocess(data_dir, cache_dir, device="cuda", artist: str | None = None):
    print(f"Scanning library at: {data_dir}")
    print(f"Cache location: {cache_dir}")
    
    manager = LibraryManager(data_dir, cache_dir)
    extractor = MuQAudioExtractor(device=device)

    # Find all artists in music/ dir
    artist_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if artist:
        artist_dirs = [a for a in artist_dirs if a == artist]
    
    if not artist_dirs:
        print("No artist directories found.")
        return

    for artist_name in artist_dirs:
        print(f"\nProcessing Artist: {artist_name}")
        career = manager.scan_artist(artist_name)
        if not career.tracks:
            print(f"No tracks found for {artist_name}. Skipping.")
            continue
        
        extractor.process_career(career)
        manager.save_to_cache(career)

    print("\nPre-processing complete.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subproject_dir = os.path.dirname(script_dir)
    default_data_dir = os.path.join(subproject_dir, "data", "music")
    default_cache_dir = os.path.join(subproject_dir, "data", "cache")

    parser = argparse.ArgumentParser(description="Extract MuQ-MuLan audio embeddings")
    parser.add_argument("--data_dir", default=default_data_dir, help="Path to music library")
    parser.add_argument("--cache_dir", default=default_cache_dir, help="Path to output cache")
    parser.add_argument("--device", default="cuda", help="Device to run model on (cuda/cpu/mps)")
    parser.add_argument("--artist", default=None, help="Optional: process a single artist directory name")
    args = parser.parse_args()
    
    preprocess(args.data_dir, args.cache_dir, args.device, artist=args.artist)

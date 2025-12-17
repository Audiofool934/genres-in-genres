"""
Library Manager Module.

Handles scanning of the music library and caching of analyzed careers.
Expected structure: data/music/{Artist}/{Year}-{Album}/*.mp3
"""
import os
import glob
import pickle
import datetime
import re
from typing import List, Optional, Dict
from .core import ArtistCareer, Track

class LibraryManager:
    """
    Manages the music library organization and cache using pickle.
    """
    
    def __init__(self, music_root: str, cache_dir: str):
        self.music_root = music_root
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)

    def scan_artist(self, artist_name: str) -> ArtistCareer:
        """
        Scans the filesystem for an artist's tracks.
        Structure: music_root/{Artist}/{Year}-{Album}/*.mp3
        """
        artist_path = os.path.join(self.music_root, artist_name)
        if not os.path.exists(artist_path):
            raise FileNotFoundError(f"Artist directory not found: {artist_path}")
            
        career = ArtistCareer(artist_name=artist_name)
        
        # Find album directories
        # Expected: 2023-AlbumName
        album_dirs = [d for d in os.listdir(artist_path) if os.path.isdir(os.path.join(artist_path, d))]
        
        for album_dir in album_dirs:
            match = re.match(r"(\d{4})-(.+)", album_dir)
            if not match:
                continue
            year = int(match.group(1))
            album_name = match.group(2)
                
            full_album_path = os.path.join(artist_path, album_dir)
            audio_files = glob.glob(os.path.join(full_album_path, "*.mp3")) + \
                          glob.glob(os.path.join(full_album_path, "*.wav"))
            
            for fpath in audio_files:
                title = os.path.splitext(os.path.basename(fpath))[0]
                # Normalize common filename patterns:
                # - "01 - Title" -> "Title"
                # - "Artist - Title" / "Artist-Title" -> "Title"
                title = re.sub(r"^\s*\d+\s*-\s*", "", title).strip()
                title = re.sub(rf"^\s*{re.escape(artist_name)}\s*-\s*", "", title).strip()
                title = re.sub(r"^\s*-\s*", "", title).strip()
                # Default date: Jan 1st of that year
                release_date = datetime.date(year, 1, 1)
                
                track = Track(
                    file_path=os.path.abspath(fpath),
                    title=title,
                    album=album_name,
                    release_date=release_date
                )
                career.add_track(track)
                
        return career

    def save_to_cache(self, career: ArtistCareer):
        """Saves the analyzed career to cache."""
        filename = f"{career.artist_name}.pkl"
        path = os.path.join(self.cache_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(career, f)
        print(f"[LibraryManager] Saved {career.artist_name} to {path}")

    def load_from_cache(self, artist_name: str) -> Optional[ArtistCareer]:
        """Loads an artist from cache."""
        filename = f"{artist_name}.pkl"
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            return None
            
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_cached_artists(self) -> List[str]:
        """Returns a list of artist names found in the cache."""
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        return [os.path.splitext(os.path.basename(f))[0] for f in files]

"""
Feature Extractor Module.

Wraps the MuQ-MuLan pipeline to extract embeddings from audio files.
"""
import torch
import torchaudio
from tqdm import tqdm
from .core import ArtistCareer, StyleEmbedding

from pipeline_mulan import MuQMuLanEncoder

class MuLanFeatureExtractor:
    """Extracts 512-dim style embeddings from audio files using MuQ-MuLan."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder = MuQMuLanEncoder(device=self.device)
        self.sample_rate = 24000

    def process_career(self, career: ArtistCareer):
        """Processes all tracks in an ArtistCareer and populates their embeddings."""
        print(f"[FeatureExtractor] Processing {len(career.tracks)} tracks for {career.artist_name}...")
        for track in tqdm(career.tracks):
            wav, sr = torchaudio.load(track.file_path)

            # Crop to middle 30s if longer
            target_len = 30 * sr
            if wav.shape[-1] > target_len:
                start = (wav.shape[-1] - target_len) // 2
                wav = wav[:, start:start+target_len]
            
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)

            with torch.no_grad():
                emb_tensor = self.encoder.encode_audio(wav, sample_rate=sr)
            
            vec = emb_tensor.cpu().numpy().squeeze(0)
            style_emb = StyleEmbedding(vector=vec, track_ref=track)
            career.add_embedding(style_emb)

        print(f"[FeatureExtractor] Done. Generated {len(career.embeddings)} embeddings.")

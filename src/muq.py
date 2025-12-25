
import torch
import numpy as np
from typing import List, Optional

class MuQTextEncoder:
    """
    Wrapper for OpenMuQ/MuQ-MuLan-large text encoder.
    Uses the official MuQ library API.
    """
    
    REPO_ID = "OpenMuQ/MuQ-MuLan-large"
    LATENT_DIM = 512

    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"[MuQTextEncoder] Initializing on {self.device}...")
        
        try:
            # Use official MuQ library API
            from muq import MuQMuLan
            
            print(f"[MuQ] Loading model from {self.REPO_ID}...")
            self.model = MuQMuLan.from_pretrained(self.REPO_ID)
            self.model = self.model.to(self.device).eval()
            
            print(f"[MuQTextEncoder] Successfully loaded model (Output dim: {self.LATENT_DIM}).")
            
        except ImportError as e:
            print(f"[MuQTextEncoder] ERROR: 'muq' library not installed. Run: pip install muq")
            raise e
        except Exception as e:
            print(f"[MuQTextEncoder] Failed to load model: {e}")
            raise e

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of text strings into embeddings.
        Returns: np.ndarray of shape [len(texts), 512] (normalized)
        """
        if not texts:
            return np.array([])
        
        with torch.no_grad():
            # Use official API: mulan(texts=texts) returns text embeddings
            embeddings = self.model(texts=texts)
            
            # Normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
        return embeddings.cpu().numpy()

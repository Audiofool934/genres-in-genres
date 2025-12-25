
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.append(os.getcwd())

try:
    from src.muq import MuQTextEncoder
    
    print("Initializing MuQTextEncoder...")
    encoder = MuQTextEncoder()
    print("Encoder initialized.")
    
    # Test cases: Antonyms should have lower similarity than Synonyms
    pairs = [
        ("Happy", "Sad"),
        ("Energetic", "Calm"),
        ("Metal", "Classical"),
        ("Love", "Hate"),
        ("Happy", "Joyful"), # Synonym
        ("Sad", "Depressed"), # Synonym
        ("Guitar", "Piano"),
        ("Techno", "Country")
    ]
    
    words = sorted(list(set([w for p in pairs for w in p])))
    print(f"Encoding {len(words)} words: {words}")
    
    embeddings = encoder.encode(words)
    word_to_vec = {w: embeddings[i] for i, w in enumerate(words)}
    
    print("\n--- Cosine Similarities ---")
    for w1, w2 in pairs:
        v1 = word_to_vec[w1].reshape(1, -1)
        v2 = word_to_vec[w2].reshape(1, -1)
        sim = cosine_similarity(v1, v2)[0][0]
        print(f"{w1} vs {w2}: {sim:.4f}")
        
    print("\n--- Sanity Check ---")
    # Check variance
    variance = np.var(embeddings)
    print(f"Embedding Variance: {variance:.6f}")
    
    if variance < 1e-5:
        print("WARNING: Low variance! Embeddings might be collapsed.")
        
except Exception as e:
    print(f"Error: {e}")

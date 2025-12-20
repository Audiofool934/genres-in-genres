"""
Feature Extractor Module (placeholder).

Runtime MuQ-MuLan extraction has been removed. The class is kept for API
compatibility and will raise a clear error if invoked.
"""


class MuLanFeatureExtractor:
    """Placeholder extractor; instructs users to rely on cached embeddings."""

    def __init__(self, device: str = "cpu"):
        raise RuntimeError(
            "MuQ-MuLan extraction has been removed from this project. "
            "Use the precomputed caches under data/cache/music or provide your own extractor."
        )

    def process_career(self, career):
        raise RuntimeError(
            "MuQ-MuLan extraction has been removed from this project. "
            "Use the precomputed caches under data/cache/music or provide your own extractor."
        )

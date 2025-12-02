"""
Utility functions: discretization, scoring, sample dataset creation
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def discretize_feature(value: float, bins: List[float], labels: List[str]):
    """
    Discretize a numeric value into labels based on bins edges.
    Example: bins=[0.4, 0.7], labels=['low','medium','high']
    """
    idx = np.digitize([value], bins=bins)[0]
    if idx >= len(labels):
        idx = len(labels) - 1
    return labels[idx]

def create_sample_tracks():
    """Create a tiny sample dataframe of tracks for demo."""
    rows = [
        {"track_id": "t1", "track_name": "Uplift", "artists": "A", "genre": "pop", "energy": 0.85, "valence": 0.9, "danceability": 0.8, "tempo": 130},
        {"track_id": "t2", "track_name": "Calm Sea", "artists": "B", "genre": "classical", "energy": 0.2, "valence": 0.3, "danceability": 0.1, "tempo": 70},
        {"track_id": "t3", "track_name": "Midday Beat", "artists": "C", "genre": "electronic", "energy": 0.6, "valence": 0.5, "danceability": 0.6, "tempo": 110},
        {"track_id": "t4", "track_name": "Focus Flow", "artists": "D", "genre": "ambient", "energy": 0.3, "valence": 0.4, "danceability": 0.2, "tempo": 75},
        {"track_id": "t5", "track_name": "Run Fast", "artists": "E", "genre": "pop", "energy": 0.95, "valence": 0.7, "danceability": 0.9, "tempo": 150},
    ]
    return pd.DataFrame(rows)

def normalize_tempo(tempo):
    # simple normalization for vector use
    return tempo / 200.0

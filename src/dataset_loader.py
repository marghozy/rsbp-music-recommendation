import pandas as pd
import numpy as np

DISCRETIZATION_SCHEMA = {
    "energy": ([0.4, 0.7], ["low", "medium", "high"]),
    "valence": ([0.4, 0.7], ["low", "medium", "high"]),
    "danceability": ([0.4, 0.7], ["low", "medium", "high"]),
    "acousticness": ([0.4, 0.7], ["low", "medium", "high"]),
    "tempo": ([90, 120], ["slow", "moderate", "fast"])
}

def discretize(value, bins, labels):
    idx = np.digitize([value], bins=bins)[0]
    if idx >= len(labels):
        idx = len(labels)-1
    return labels[idx]

def load_dataset(path: str):
    """Load your dataset and auto-discretize audio features."""
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Fix naming mismatch (track_genre → genre)
    if "track_genre" in df.columns:
        df.rename(columns={"track_genre": "genre"}, inplace=True)

    required = ["track_name", "artists", "genre", 
                "energy", "valence", "danceability", 
                "acousticness", "tempo"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Discretize
    for feature, (bins, labels) in DISCRETIZATION_SCHEMA.items():
        df[f"{feature}_cat"] = df[feature].apply(lambda v: discretize(v, bins, labels))

    # Numeric vector for CBR
    df["vector"] = df.apply(lambda row: [
        row.energy,
        row.valence,
        row.danceability,
        row.acousticness,
        row.tempo / 200.0
    ], axis=1)

    return df
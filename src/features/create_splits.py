"""
Artist-Level Split Generator for Label Audit.
Partitions DEAM, PMEmo, and Jamendo datasets by artist so that no artist appears
in more than one split (Zero Artist Overlap).
"""

import os
import re
import json
import glob
import pandas as pd
import numpy as np

RANDOM_SEED = 42

def normalize_artist(name):
    if not isinstance(name, str):
        return "unknown_artist"
    cleaned = name.lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else "unknown_artist"

def make_artist_splits(items_df, artist_col="artist", id_col="id", train_ratio=0.70, val_ratio=0.15, seed=RANDOM_SEED):
    """
    Partitions items strictly by artist group into train, val, and test splits.
    """
    np.random.seed(seed)
    items_df = items_df.copy()
    items_df["norm_artist"] = items_df[artist_col].apply(normalize_artist)

    # Group tracks by artist
    artist_to_ids = {}
    for _, row in items_df.iterrows():
        a = row["norm_artist"]
        tid = row[id_col]
        if a not in artist_to_ids:
            artist_to_ids[a] = []
        artist_to_ids[a].append(tid)

    unique_artists = sorted(list(artist_to_ids.keys()))
    np.random.shuffle(unique_artists)

    n_artists = len(unique_artists)
    n_train = int(n_artists * train_ratio)
    n_val = int(n_artists * val_ratio)

    train_artists = set(unique_artists[:n_train])
    val_artists = set(unique_artists[n_train:n_train + n_val])
    test_artists = set(unique_artists[n_train + n_val:])

    # Strict assertion: Zero artist leakage
    assert train_artists.isdisjoint(val_artists), "Artist leakage detected between train and val!"
    assert train_artists.isdisjoint(test_artists), "Artist leakage detected between train and test!"
    assert val_artists.isdisjoint(test_artists), "Artist leakage detected between val and test!"

    train_ids = [tid for a in train_artists for tid in artist_to_ids[a]]
    val_ids = [tid for a in val_artists for tid in artist_to_ids[a]]
    test_ids = [tid for a in test_artists for tid in artist_to_ids[a]]

    return {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "train_artists": sorted(list(train_artists)),
        "val_artists": sorted(list(val_artists)),
        "test_artists": sorted(list(test_artists)),
    }

def build_deam_pmemo_splits(output_path="data/splits/splits_deam_pmemo.json"):
    records = []

    # 1. DEAM metadata
    meta1 = pd.read_csv("data/raw/deam/metadata/metadata_2013.csv", on_bad_lines="skip")
    meta2 = pd.read_csv("data/raw/deam/metadata/metadata_2014.csv", on_bad_lines="skip")
    meta3 = pd.read_csv("data/raw/deam/metadata/metadata_2015.csv", on_bad_lines="skip")

    for _, row in meta1.iterrows():
        sid = f"deam_{row['song_id']}"
        artist = str(row.get("Artist", "deam_artist_unknown"))
        records.append({"id": sid, "dataset": "deam", "artist": artist})

    for _, row in meta2.iterrows():
        sid = f"deam_{row['Id']}"
        artist = str(row.get("Artist", "deam_artist_unknown"))
        records.append({"id": sid, "dataset": "deam", "artist": artist})

    for _, row in meta3.iterrows():
        sid = f"deam_{row.iloc[0]}"
        artist = str(row.iloc[2]) if len(row) > 2 else "deam_artist_unknown"
        records.append({"id": sid, "dataset": "deam", "artist": artist})

    # 2. PMEmo metadata
    pmemo_meta = pd.read_csv("data/raw/pmemo/PMEmo2019/metadata.csv")
    for _, row in pmemo_meta.iterrows():
        sid = f"pmemo_{row['musicId']}"
        artist = str(row.get("artist", "pmemo_artist_unknown"))
        records.append({"id": sid, "dataset": "pmemo", "artist": artist})

    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    print(f"Total DEAM + PMEmo tracks to split: {len(df)}")

    splits = make_artist_splits(df, artist_col="artist", id_col="id")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"DEAM + PMEmo splits saved to {output_path}")
    print(f"Train tracks: {len(splits['train_ids'])}, Val tracks: {len(splits['val_ids'])}, Test tracks: {len(splits['test_ids'])}")
    print(f"Train artists: {len(splits['train_artists'])}, Val artists: {len(splits['val_artists'])}, Test artists: {len(splits['test_artists'])}")
    return splits

def build_jamendo_splits(output_path="data/splits/splits_jamendo.json"):
    # Load tracks present on disk in data/raw/mtg-jamendo/
    audio_files = glob.glob("data/raw/mtg-jamendo/*/*.mp3")
    local_paths = {os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f).replace(".low.mp3", ".mp3") for f in audio_files}

    # Match against split-0 TSV files
    train_tsv = "data/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv"
    val_tsv = "data/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv"
    test_tsv = "data/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv"

    def read_jamendo_tsv(tsv_file):
        tracks = []
        with open(tsv_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    tid, aid, albid, path, dur = parts[:5]
                    tags = parts[5:]
                    tracks.append({"id": tid, "artist": aid, "path": path, "tags": tags})
        return tracks

    train_data = read_jamendo_tsv(train_tsv)
    val_data = read_jamendo_tsv(val_tsv)
    test_data = read_jamendo_tsv(test_tsv)

    # Filter to only local downloaded tracks
    train_ids = [t["id"] for t in train_data if t["path"] in local_paths]
    val_ids = [t["id"] for t in val_data if t["path"] in local_paths]
    test_ids = [t["id"] for t in test_data if t["path"] in local_paths]

    train_artists = {t["artist"] for t in train_data if t["id"] in set(train_ids)}
    val_artists = {t["artist"] for t in val_data if t["id"] in set(val_ids)}
    test_artists = {t["artist"] for t in test_data if t["id"] in set(test_ids)}

    # Assert zero artist leakage
    assert train_artists.isdisjoint(val_artists), "Jamendo train/val artist leakage!"
    assert train_artists.isdisjoint(test_artists), "Jamendo train/test artist leakage!"
    assert val_artists.isdisjoint(test_artists), "Jamendo val/test artist leakage!"

    splits = {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "train_artists": sorted(list(train_artists)),
        "val_artists": sorted(list(val_artists)),
        "test_artists": sorted(list(test_artists)),
    }

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Jamendo splits saved to {output_path}")
    print(f"Train tracks: {len(splits['train_ids'])}, Val tracks: {len(splits['val_ids'])}, Test tracks: {len(splits['test_ids'])}")
    print(f"Train artists: {len(splits['train_artists'])}, Val artists: {len(splits['val_artists'])}, Test artists: {len(splits['test_artists'])}")
    return splits

if __name__ == "__main__":
    build_deam_pmemo_splits()
    build_jamendo_splits()

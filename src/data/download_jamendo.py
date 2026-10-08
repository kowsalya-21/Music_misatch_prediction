"""
Official MTG-Jamendo dataset downloader with multi-threaded download,
SHA256 checksum verification for archives and unpacked audio tracks.
Based on the official MTG repository scripts.
"""

import os
import csv
import hashlib
import tarfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

MTG_BASE = Path("data/mtg-jamendo-dataset")
ID_FILE_PATH = MTG_BASE / "data/download"
MIRROR_BASE = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low"

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_file(url, target_path, expected_sha256):
    if os.path.exists(target_path):
        if compute_sha256(target_path) == expected_sha256:
            print(f"Skipping {os.path.basename(target_path)} (already exists and checksum matches)")
            return target_path
        else:
            print(f"Checksum mismatch for existing {target_path}, redownloading...")
            os.remove(target_path)

    temp_path = f"{target_path}.tmp"
    res = requests.get(url, stream=True, timeout=60)
    res.raise_for_status()
    total_size = int(res.headers.get("content-length", 0))

    with open(temp_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=os.path.basename(target_path), leave=False
    ) as pbar:
        for chunk in res.iter_content(chunk_size=1024 * 512):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    actual_sha256 = compute_sha256(temp_path)
    if actual_sha256 != expected_sha256:
        os.remove(temp_path)
        raise ValueError(f"Checksum mismatch for {target_path}: expected {expected_sha256}, got {actual_sha256}")

    os.rename(temp_path, target_path)
    return target_path

def download_and_unpack_jamendo(output_dir="data/raw/mtg-jamendo", num_tars=10, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    tars_file = ID_FILE_PATH / "autotagging_moodtheme_audio-low_sha256_tars.txt"
    tracks_file = ID_FILE_PATH / "autotagging_moodtheme_audio-low_sha256_tracks.txt"

    with open(tars_file) as f:
        sha256_tars = dict((row[1], row[0]) for row in csv.reader(f, delimiter=" ") if len(row) == 2)

    with open(tracks_file) as f:
        sha256_tracks = dict((row[1], row[0]) for row in csv.reader(f, delimiter=" ") if len(row) == 2)

    all_tars = sorted(list(sha256_tars.keys()))[:num_tars]
    print(f"Downloading {len(all_tars)} MTG-Jamendo archives into {output_dir}...")

    # Parallel download
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for tar_name in all_tars:
            url = f"{MIRROR_BASE}/{tar_name}"
            target = os.path.join(output_dir, tar_name)
            expected = sha256_tars[tar_name]
            futures[executor.submit(download_file, url, target, expected)] = tar_name

        for future in as_completed(futures):
            tar_name = futures[future]
            try:
                future.result()
                print(f"[OK] Downloaded & verified tar: {tar_name}")
            except Exception as e:
                print(f"[ERROR] Failed {tar_name}: {e}")
                raise e

    # Unpack and verify tracks
    print("Unpacking archives and verifying track SHA256 checksums...")
    verified_tracks = 0
    for tar_name in all_tars:
        tar_path = os.path.join(output_dir, tar_name)
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            tar.extractall(path=output_dir)

            for member in members:
                track_path = os.path.join(output_dir, member.name)
                expected_track_sha = sha256_tracks.get(member.name)
                if expected_track_sha:
                    actual_track_sha = compute_sha256(track_path)
                    assert actual_track_sha == expected_track_sha, f"Corrupted track: {member.name}"
                    verified_tracks += 1

        # Remove tar archive after extraction to save disk
        os.remove(tar_path)

    print(f"Successfully unpacked and verified {verified_tracks} MTG-Jamendo tracks!")
    return verified_tracks

if __name__ == "__main__":
    download_and_unpack_jamendo()

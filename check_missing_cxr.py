"""Check how many rows in the wide parquet have a CXR jpg that doesn't exist on disk.

Usage:
    python check_missing_cxr.py \\
        --features /path/to/mimic_cxr_features.parquet \\
        --cxr-root /path/to/mimic-cxr-jpg-2.1.0.physionet.org

    # Also save the missing rows to a separate parquet for investigation:
    python check_missing_cxr.py ... --save-missing missing.parquet

Output:
    Total rows           : 116,129
    With cxr_path column : 0
    Reconstructed paths  : 116,129
    Existing on disk     : 113,842  (98.0%)
    Missing on disk      :   2,287  ( 2.0%)
    Top 10 missing patterns by subject_id prefix:
        p10: 234
        p11: 198
        ...
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def cxr_path(root, subject_id, study_id, dicom_id):
    """MIMIC-CXR-JPG layout: files/p<XX>/p<subject>/s<study>/<dicom>.jpg"""
    p_top = f"p{str(subject_id)[:2]}"
    return Path(root) / "mimic" / p_top / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def check_one(row, cxr_root):
    """Return (path_str, exists_bool)."""
    # Prefer existing cxr_path column if present, else reconstruct
    if "cxr_path" in row and isinstance(row.cxr_path, str) and row.cxr_path:
        p = Path(row.cxr_path)
    else:
        p = cxr_path(cxr_root, int(row.subject_id), int(row.study_id),
                     str(row.dicom_id))
    return str(p), p.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="mimic_cxr_features.parquet from Data.ipynb")
    ap.add_argument("--cxr-root", required=True,
                    help="MIMIC-CXR-JPG root dir")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel threads for file existence check (IO-bound)")
    ap.add_argument("--save-missing", default=None,
                    help="optional: save missing rows to this parquet")
    args = ap.parse_args()

    print(f"loading {args.features} ...", flush=True)
    df = pd.read_parquet(args.features)
    n = len(df)
    print(f"  rows: {n:,}", flush=True)

    has_path_col = "cxr_path" in df.columns
    if has_path_col:
        n_with_path = int(df["cxr_path"].notna().sum())
        print(f"  rows with cxr_path column: {n_with_path:,}", flush=True)
    else:
        n_with_path = 0
        print("  (no cxr_path column — will reconstruct paths from ids)", flush=True)

    # ---- parallel existence check ----
    print(f"checking {n:,} files on disk (workers={args.workers}) ...", flush=True)
    rows = list(df.itertuples(index=False))
    results = [None] * n
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(tqdm(
                pool.map(lambda r: check_one(r, args.cxr_root), rows),
                total=n)):
            results[i] = res

    df["_path"] = [r[0] for r in results]
    df["_exists"] = [r[1] for r in results]

    n_exists = int(df["_exists"].sum())
    n_missing = n - n_exists

    # ---- summary ----
    print()
    print("=" * 60)
    print(f"Total rows           : {n:,}")
    print(f"With cxr_path column : {n_with_path:,}")
    print(f"Reconstructed paths  : {n - n_with_path:,}")
    print(f"Existing on disk     : {n_exists:,}  ({100*n_exists/n:.1f}%)")
    print(f"Missing on disk      : {n_missing:,}  ({100*n_missing/n:.1f}%)")
    print("=" * 60)

    if n_missing > 0:
        # Group missing by subject_id 2-digit prefix
        missing = df[~df["_exists"]].copy()
        if "subject_id" in missing.columns:
            missing["_prefix"] = missing["subject_id"].astype(str).str[:2].apply(lambda s: f"p{s}")
            top = Counter(missing["_prefix"]).most_common(10)
            print()
            print("Top 10 missing patterns by subject_id prefix:")
            for k, v in top:
                print(f"  {k}: {v:,}")

        # Show a few example missing paths
        print()
        print("First 5 missing paths:")
        for p in missing["_path"].head(5):
            print(f"  {p}")

        if args.save_missing:
            out = Path(args.save_missing)
            out.parent.mkdir(parents=True, exist_ok=True)
            missing.drop(columns=["_prefix"], errors="ignore") \
                   .to_parquet(out, index=False)
            print()
            print(f"saved {len(missing):,} missing rows → {out}")


if __name__ == "__main__":
    main()

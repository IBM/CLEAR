#!/usr/bin/env python3
"""
Filter and sample CSV files based on traj_score column.
For each subdirectory containing direct CSVs in /Users/lilache/PycharmProjects/CLEAR/dev/runs:
- Sample 10 CSVs with score > 0.9 (good)
- Sample 10 CSVs with score < 0.26 (bad)
- Save in mirrored folder structure under runs_filtered
"""

import random
from pathlib import Path
from collections import defaultdict
import shutil
import pandas as pd


def get_traj_score(csv_path):
    """Read the traj_score value from the first data row of a CSV file."""
    try:
        # Read only the first row and the traj_score column
        df = pd.read_csv(csv_path, nrows=1, usecols=['traj_score'])
        if not df.empty and 'traj_score' in df.columns:
            return float(df['traj_score'].iloc[0])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None


def main():
    base_path = Path('/Users/lilache/PycharmProjects/CLEAR/dev/runs/input_all')
    output_base = Path('/Users/lilache/PycharmProjects/CLEAR/dev/runs/input')

    # Find all subdirectories that contain CSV files directly
    subdirs_with_csvs = defaultdict(lambda: {'good': [], 'bad': []})

    print("Scanning for CSV files...")
    for subdir in base_path.rglob('*'):
        if not subdir.is_dir():
            continue

        # Get CSV files directly in this directory (not in subdirectories)
        csv_files = list(subdir.glob('*.csv'))

        if not csv_files:
            continue

        # Get relative path from base
        rel_path = subdir.relative_to(base_path)

        print(f"\nProcessing {rel_path} ({len(csv_files)} CSV files)...")

        # Categorize CSVs by score
        for csv_file in csv_files:
            score = get_traj_score(csv_file)
            if score is None:
                continue

            if score > 0.9:
                subdirs_with_csvs[rel_path]['good'].append((csv_file, score))
            elif score < 0.26:
                subdirs_with_csvs[rel_path]['bad'].append((csv_file, score))

    # Sample and copy files
    print("\n" + "=" * 60)
    print("Sampling and copying files...")
    print("=" * 60)

    for rel_path, categories in subdirs_with_csvs.items():
        good_files = categories['good']
        bad_files = categories['bad']

        print(f"\n{rel_path}:")
        print(f"  Found {len(good_files)} good CSVs (score > 0.9)")
        print(f"  Found {len(bad_files)} bad CSVs (score < 0.26)")

        # Sample up to 10 from each category
        good_sample = random.sample(good_files, min(10, len(good_files)))
        bad_sample = random.sample(bad_files, min(10, len(bad_files)))

        # Copy good files
        if good_sample:
            good_output_dir = output_base / rel_path / 'good'
            good_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Copying {len(good_sample)} good files to {good_output_dir}")
            for csv_file, score in good_sample:
                dest = good_output_dir / csv_file.name
                shutil.copy2(csv_file, dest)
                print(f"    {csv_file.name} (score: {score:.4f})")

        # Copy bad files
        if bad_sample:
            bad_output_dir = output_base / rel_path / 'bad'
            bad_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Copying {len(bad_sample)} bad files to {bad_output_dir}")
            for csv_file, score in bad_sample:
                dest = bad_output_dir / csv_file.name
                shutil.copy2(csv_file, dest)
                print(f"    {csv_file.name} (score: {score:.4f})")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

# Made with Bob

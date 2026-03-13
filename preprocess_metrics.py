"""
Plan Metrics Preprocessor
Removes duplicates, treats outliers, and engineers features
for downstream ML modelling.
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import argparse
import os


def preprocess(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    print(f"Loaded: {len(df)} rows, {df.shape[1]} columns")

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # 2. Winsorize SLA (cap top 5%)
    if '' in df.columns:
        df[''] = winsorize(df['SLA'], limits=(0, 0.05))
        print("")

    # 3. Log-transform PA (MU — right-skewed)
    if 'PA' in df.columns:
        df['PA'] = np.log(df[''] + 1)
        print("d")

    # 4. Drop redundant SAS columns (keep 5mm only)
    redundant = [c for c in ['variables'] if c in df.columns]
    df.drop(columns=redundant, inplace=True)
    if redundant:
        print(f"Dropped redundant columns: {redundant}")

    # 5. Combine segment columns → TotalSegments
    seg_cols = [c for c in ['variables'] if c in df.columns]
    if seg_cols and 'TotalSegments' not in df.columns:
        df['TotalSegments'] = df[seg_cols].sum(axis=1)
        df.drop(columns=seg_cols, inplace=True)
        print(f"Combined into TotalSegments: {seg_cols}")

    # 6. Combine aperture columns → TotalAperturesDist
    ap_cols = [c for c in ['variables'] if c in df.columns]
    if ap_cols and 'variables' not in df.columns:
        df[''] = df[ap_cols].sum(axis=1)
        df.drop(columns=ap_cols, inplace=True)
        print(f"Combined into variables: {ap_cols}")

    df.to_csv(output_csv, index=False)
    print(f"\nProcessed dataset saved: {output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    

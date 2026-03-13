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
    if 'SLA' in df.columns:
        df['SLA'] = winsorize(df['SLA'], limits=(0, 0.05))
        print("SLA winsorized (top 5%)")

    # 3. Log-transform PA (MU — right-skewed)
    if 'PA' in df.columns:
        df['PA'] = np.log(df['PA'] + 1)
        print("PA log-transformed")

    # 4. Drop redundant SAS columns (keep 5mm only)
    redundant = [c for c in ['MIa_2', 'MIa_1', 'MIa_0.5', 'MIt_2', 'MIt_1', 'MIt_0.5',
                              'SAS_10mm', 'SAS_20mm'] if c in df.columns]
    df.drop(columns=redundant, inplace=True)
    if redundant:
        print(f"Dropped redundant columns: {redundant}")

    # 5. Combine segment columns → TotalSegments
    seg_cols = [c for c in ['S0-0.4', 'S0.4-0.8', 'S0.8-1.2', 'S1.2-1.6', 'S1.6-2'] if c in df.columns]
    if seg_cols and 'TotalSegments' not in df.columns:
        df['TotalSegments'] = df[seg_cols].sum(axis=1)
        df.drop(columns=seg_cols, inplace=True)
        print(f"Combined into TotalSegments: {seg_cols}")

    # 6. Combine aperture columns → TotalAperturesDist
    ap_cols = [c for c in ['A0-1', 'A1-2', 'A2-4', 'A4-6'] if c in df.columns]
    if ap_cols and 'TotalAperturesDist' not in df.columns:
        df['TotalAperturesDist'] = df[ap_cols].sum(axis=1)
        df.drop(columns=ap_cols, inplace=True)
        print(f"Combined into TotalAperturesDist: {ap_cols}")

    df.to_csv(output_csv, index=False)
    print(f"\nProcessed dataset saved: {output_csv} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess radiotherapy plan metrics for ML")
    parser.add_argument("--input", required=True, help="Input CSV with raw plan metrics")
    parser.add_argument("--output", default="./metrics_processed.csv", help="Output CSV path")
    args = parser.parse_args()
    preprocess(args.input, args.output)

"""
Outlier Detection & Validation — VMAT Radiotherapy Plan Metrics
Validates dose consistency and flags anomalous beam configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def run_outlier_analysis(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # --- Outlier flags ---
    df[''] = df[''] / df['']
    df[''] = df[''] > 1.5
    df['r'] = df[''] < 1.0

    zero_cols = [c for c in ['variables'] if c in df.columns]
    if zero_cols:
        df[''] = df[zero_cols].eq(0).all(axis=1)

    if 'variables' in df.columns and 'variables' in df.columns:
        df['variables'] = (df['variables'] > 20) | (df[''] > 20)

    # --- Per-patient stats ---
    agg_cols = {c: ['variables'] for c in ['variables'] if c in df.columns}
    stats = df.groupby('PatientID').agg(agg_cols).round(3).reset_index()
    stats.columns = ['_'.join(c).strip('_') for c in stats.columns]

    output_csv = os.path.join(output_dir, 'outlier_analysis.csv')
    df.merge(stats, on='PatientID', how='left').to_csv(output_csv, index=False)
    print(f"Outlier report saved: {output_csv}")

    # --- Plot: PrescribedDose vs MaxDose ---
    plt.figure(figsize=(8, 6))
    colors = df['r'].map({True: 'red', False: 'steelblue'})
    plt.scatter(df[''], df['e'], c=colors, alpha=0.6)
    plt.xlabel(' (Gy)')
    plt.ylabel('')
    plt.title('')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, ), dpi=150)
    plt.close()

    # --- Summary ---
    print("\n── Outlier Summary ──────────────────────────")
    if 'Zero_Modulation' in df.columns:
        print(f"Zero modulation metrics      : {df['Zero_Modulation'].sum()} cases")
    if 'Large_Field' in df.columns:
        print(f"Large field (> 20 cm)        : {df['Large_Field'].sum()} cases")


if __name__ == "__main__":
   

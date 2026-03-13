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
    df['DoseRatio'] = df['MaxDose'] / df['PrescribedDose']
    df['MaxDose_Outlier'] = df['DoseRatio'] > 1.5
    df['PrescribedDose_Outlier'] = df['PrescribedDose'] < 1.0

    zero_cols = [c for c in ['MI_s_2', 'MI_s_1', 'MI_a_2', 'MI_a_1'] if c in df.columns]
    if zero_cols:
        df['Zero_Modulation'] = df[zero_cols].eq(0).all(axis=1)

    if 'Field_X_1' in df.columns and 'Field_Y_1' in df.columns:
        df['Large_Field'] = (df['Field_X_1'] > 20) | (df['Field_Y_1'] > 20)

    # --- Per-patient stats ---
    agg_cols = {c: ['mean', 'std'] for c in ['MCS', 'SAS_5mm', 'MaxDose', 'PrescribedDose'] if c in df.columns}
    stats = df.groupby('PatientID').agg(agg_cols).round(3).reset_index()
    stats.columns = ['_'.join(c).strip('_') for c in stats.columns]

    output_csv = os.path.join(output_dir, 'outlier_analysis.csv')
    df.merge(stats, on='PatientID', how='left').to_csv(output_csv, index=False)
    print(f"Outlier report saved: {output_csv}")

    # --- Plot: PrescribedDose vs MaxDose ---
    plt.figure(figsize=(8, 6))
    colors = df['MaxDose_Outlier'].map({True: 'red', False: 'steelblue'})
    plt.scatter(df['PrescribedDose'], df['MaxDose'], c=colors, alpha=0.6)
    plt.xlabel('Prescribed Dose (Gy)')
    plt.ylabel('Max Dose (Gy)')
    plt.title('Prescribed vs Max Dose  (red = outlier > 150%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prescribed_vs_maxdose.png'), dpi=150)
    plt.close()

    # --- Summary ---
    print("\n── Outlier Summary ──────────────────────────")
    print(f"MaxDose > 150% of Prescribed : {df['MaxDose_Outlier'].sum()} cases")
    print(f"PrescribedDose < 1 Gy        : {df['PrescribedDose_Outlier'].sum()} cases")
    if 'Zero_Modulation' in df.columns:
        print(f"Zero modulation metrics      : {df['Zero_Modulation'].sum()} cases")
    if 'Large_Field' in df.columns:
        print(f"Large field (> 20 cm)        : {df['Large_Field'].sum()} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier detection for radiotherapy plan metrics")
    parser.add_argument("--csv", required=True, help="Path to the metrics CSV file")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()
    run_outlier_analysis(args.csv, args.output)

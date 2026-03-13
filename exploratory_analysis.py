"""
Exploratory Data Analysis — VMAT Radiotherapy Plans
Generates descriptive statistics and visualizations for treatment plan metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

sns.set_style('whitegrid')
sns.set_context("notebook", font_scale=1.2)


def run_analysis(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Numeric columns only (exclude identifiers)
    exclude = ['PatientID', 'Feixe', 'RTPlan']
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    df_num = df[numeric_cols]

    # Descriptive statistics
    stats = df_num.describe(percentiles=[0.25, 0.5, 0.75]).round(4)
    modes = df_num.mode().iloc[0].rename('mode').round(4)
    stats = pd.concat([stats, modes.to_frame().T])

    stats_path = os.path.join(output_dir, 'descriptive_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("")
        f.write(stats.to_string())
    print(f"Stats saved: {stats_path}")

    # Plot 1: ALS distribution
    if 'ALS' in df_num.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_num['ALS'], bins=20, color='steelblue', edgecolor='black', kde=True)
        plt.axvline(df_num['ALS'].mean(), color='red', linestyle='--', label=f"Mean: {df_num['ALS'].mean():.4f}")
        plt.axvline(df_num['ALS'].median(), color='green', linestyle='--', label=f"Median: {df_num['ALS'].median():.4f}")
        plt.title('Leaf Speed Distribution (ALS)')
        plt.xlabel('ALS (cm/s)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_als.png'), dpi=150)
        plt.close()

    # Plot 2: SAS boxplot
    sas_cols = [c for c in ['SAS_5mm', 'SAS_10mm', 'SAS_20mm'] if c in df_num.columns]
    if sas_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_num[sas_cols], palette='Set2')
        sns.swarmplot(data=df_num[sas_cols], color='black', size=4, alpha=0.5)
        plt.title('Small Aperture Score (SAS) Distribution')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplot_sas.png'), dpi=150)
        plt.close()

    # Plot 3: MCS vs MAD
    if 'MCS' in df_num.columns and 'MAD' in df_num.columns:
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(df_num['MCS'], df_num['MAD'],
                         c=df_num.get('PrescribedDose', pd.Series(np.zeros(len(df_num)))),
                         cmap='plasma', s=80, alpha=0.7)
        plt.colorbar(sc, label='Prescribed Dose (Gy)')
        plt.title('MCS vs MAD (colored by Prescribed Dose)')
        plt.xlabel('MCS — Modulation Complexity Score')
        plt.ylabel('MAD — Mean Aperture Displacement (cm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_mcs_mad.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    

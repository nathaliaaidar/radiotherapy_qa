"""
RT Dose Extractor
Reads RTDOSE DICOM files and exports dose statistics to CSV.
"""

import pydicom
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path


def extract_dose(file_path: str) -> dict | None:
    try:
        ds = pydicom.dcmread(file_path)
        if ds.Modality != 'RTDOSE':
            print(f"Skipping {file_path} — not RTDOSE (got {ds.Modality})")
            return None

        dose = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
      
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_folder(root_dir: str, output_csv: str):
    results = []
    for dcm_file in Path(root_dir).rglob("RD*.dcm"):
        info = extract_dose(str(dcm_file))
        if info:
            results.append(info)
            print(f"✔ {dcm_file.parent.name} — Max: {info['']:.4f} Gy")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n{len(results)} files processed. Saved to: {output_csv}")
    else:
        print("No valid RTDOSE files found.")


if __name__ == "__main__":
   

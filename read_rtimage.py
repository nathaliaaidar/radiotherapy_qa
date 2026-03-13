"""
DICOM RT Image Reader
Reads and displays basic metadata and pixel statistics from RTIMAGE files.
"""

import pydicom
import numpy as np
import argparse
import os


def read_rtimage(file_path: str, show_plot: bool = False):
    if not os.path.exists(file_path):
        print(f"Error: file not found — {file_path}")
        return None

    dcm = pydicom.dcmread(file_path)

    if dcm.Modality != 'RTIMAGE':
        print(f"Warning: expected RTIMAGE, got {dcm.Modality}")

    pixel_array = dcm.pixel_array
    slope = float(getattr(dcm, 'RescaleSlope', 1.0))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
    calibrated = pixel_array * slope + intercept

    info = {
        'Modality': dcm.Modality,
        'RTImageLabel': getattr(dcm, 'RTImageLabel', 'N/A'),
        'GantryAngle': getattr(dcm, 'GantryAngle', 'N/A'),
        'ImagePlanePixelSpacing': getattr(dcm, 'ImagePlanePixelSpacing', 'N/A'),
        'RescaleSlope': slope,
        'RescaleIntercept': intercept,
        'Shape': pixel_array.shape,
        'PixelMin': float(np.min(calibrated)),
        'PixelMax': float(np.max(calibrated)),
        'PixelMean': float(np.mean(calibrated)),
    }

    print("\n── RTIMAGE Metadata ─────────────────────────")
    for k, v in info.items():
        print(f"  {k:<28} {v}")

    if show_plot:
        import matplotlib.pyplot as plt
        plt.imshow(calibrated, cmap='gray')
        plt.title(f"RTIMAGE — {info['RTImageLabel']}")
        plt.colorbar(label='Calibrated Units')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and inspect a DICOM RTIMAGE file")
    parser.add_argument("file", help="Path to the .dcm RTIMAGE file")
    parser.add_argument("--plot", action="store_true", help="Display the image")
    args = parser.parse_args()
    read_rtimage(args.file, show_plot=args.plot)

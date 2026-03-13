# ☢️ Radiotherapy QA — Data Science & DICOM Processing

Research codebase for the MSc dissertation at **IPEN-CNEN/USP**, focused on applying Data Science and Machine Learning to radiotherapy quality assurance (QA) for VMAT prostate plans.

> ⚠️ **Note:** Core metric extraction and gamma index calculation scripts are not included in this public repository as they are part of ongoing academic research.

---

## 🎯 Research Context

Portal dosimetry QA verifies that radiation doses delivered to patients match the planned doses. This project builds a data pipeline to:

1. Extract and process DICOM files (RTPLAN, RTDOSE, RTIMAGE)
2. Compute plan complexity metrics (MCS, SAS, MAD, ALS, etc.)
3. Train predictive models to anticipate gamma index passing rates **before** physical measurement
4. Reduce unnecessary QA measurements while maintaining patient safety

The dataset consists of **66 prostate VMAT plans** from a Brazilian radiation oncology center.

---

## 🛠️ Tech Stack

| Area | Technologies |
|------|-------------|
| DICOM Processing | PyDICOM, PyMedPhys |
| Data Analysis | Pandas, NumPy, SciPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest, cross-validation, SMOTE) |
| Statistical Modelling | Statsmodels (Poisson regression with Lasso) |
| Environment | Python 3.10+, Jupyter, Linux |

---

## 📁 Repository Structure

```
radiotherapy-qa-ml/
│
├── src/
│   ├── dicom/
│   │   ├── read_rtimage.py          # Read and inspect RTIMAGE DICOM files
│   │   └── extract_rtdose.py        # Batch dose extraction from RTDOSE files
│   │
│   ├── analysis/
│   │   ├── exploratory_analysis.py  # Descriptive stats + visualizations
│   │   └── outlier_detection.py     # Outlier flagging and validation
│   │
│   └── preprocessing/
│       └── preprocess_metrics.py    # Feature engineering pipeline
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/nathaliaaidar/radiotherapy-qa-ml.git
cd radiotherapy-qa-ml
pip install -r requirements.txt
```

---

## 🚀 Usage

### Read a DICOM RTIMAGE file
```bash
python src/dicom/read_rtimage.py path/to/file.dcm --plot
```

### Batch-extract dose from a patient folder
```bash
python src/dicom/extract_rtdose.py --dir ./data/patients --output ./output/dose_summary.csv
```

### Exploratory analysis
```bash
python src/analysis/exploratory_analysis.py --csv ./data/metrics.csv --output ./output/plots
```

### Outlier detection
```bash
python src/analysis/outlier_detection.py --csv ./data/metrics.csv --output ./output
```

### Preprocess metrics for ML
```bash
python src/preprocessing/preprocess_metrics.py --input ./data/metrics.csv --output ./output/metrics_processed.csv
```

---

## 📊 Plan Complexity Metrics

| Metric | Description |
|--------|-------------|
| MCS | Modulation Complexity Score |
| SAS_5mm | Small Aperture Score (< 5 mm) |
| MAD | Mean Aperture Displacement |
| ALS | Average Leaf Speed |
| SLS | Leaf Speed Standard Deviation |
| PA | Total Monitor Units (MU) |

---

## 📚 References

- Valdes G. et al. — *A mathematical framework for virtual patient-specific QA* (Med. Phys., 2016)
- TG-218 — *Tolerance limits and methodologies for IMRT measurement-based verification QA* (AAPM, 2018)
- PyMedPhys — open-source library for medical physics

---

## 👩‍💻 Author

**Nathalia Luzia Aidar Alves**
MSc Candidate — Nuclear Technology (Medical Physics / Data Science)
IPEN-CNEN/USP — São Paulo, Brazil

# 🛡️ TeleShield — GeoAI Telecom Tower Flood Resiliency System
### ASEAN GeoAI Fusion 2026 Hackathon (Telecommunications Challenge Domain, ESG Theme)

TeleShield is an end-to-end GeoAI and machine learning ensemble system engineered to predict flood risks for telecommunications cell towers across **Klang Valley, Selangor, Malaysia** (`[101.30°E, 2.85°N, 101.85°E, 3.35°N]`). 

The system validates its predictions against real historical ground-truth records from the **Department of Irrigation and Drainage (DID) Malaysia** (specifically the catastrophic December 2021 Klang Valley floods), explains individual predictions via **SHAP**, audits sub-regional fairness between Urban Core and Suburban Fringe districts, provides human-in-the-loop SQLite feedback with on-demand model retraining, and refreshes daily via an automated pipeline.

---

## 📌 Executive Summary & Key Highlights
* **Pilot Region**: Klang Valley, Selangor (Shah Alam, Klang, Petaling Jaya, Subang Jaya, Kuala Lumpur, Hulu Langat, Sepang/Dengkil).
* **Target Benchmark**: $\ge85.0\%$ classification agreement with recorded DID historical flood incidents (Achieved: **86.75%** baseline, **87.95%** post-human feedback retraining).
* **Zero Synthetic Data Compliance**: 100% of data is derived from official, traceable public datasets (OpenCelliD Malaysia MCC 502, SRTM DEM 30m, JRC Global Surface Water, Sentinel-2, CHIRPS Daily Rainfall, DID Dec 2021 reports).
* **OpenCelliD Attribution**: *"Data from OpenCelliD licensed under CC BY-SA 4.0"*.
* **Fairness & ESG Governance**: Automated bias auditing between high-density `Urban_Core` and peripheral `Suburban_Fringe` districts.

---

## 🏗️ Technical Architecture & System Modules

```
d:\AAAA\1\s\
├── config/
│   └── config.py                 # Bounds, paths, district lookup, GEE metadata, feature lists
├── data/
│   ├── raw/                      # OpenCelliD Malaysia raw tower export & DID flood points
│   ├── processed/                # Extracted feature vectors, spatial split sets, scored predictions
│   ├── cache/                    # GEE/CHIRPS cached rasters and daily refresh status logs
│   └── teleshield_feedback.db    # SQLite database for human-in-the-loop risk overrides
├── models/
│   ├── baseline_rf.joblib        # Trained Random Forest classifier
│   └── xgboost_model.joblib      # Trained XGBoost classifier
├── src/
│   ├── modules/
│   │   ├── ingestion.py          # GEE spatial layers & OpenCelliD tower loader with regional tags
│   │   ├── spatial_labels.py     # DID ground truth labeling & district spatial split (zero leakage)
│   │   ├── ml_engine.py          # RF & XGBoost classifiers, threshold calibration & DID validation
│   │   ├── explainability.py     # SHAP TreeExplainer & natural language factor attributions
│   │   ├── feedback_loop.py      # SQLite override logger & on-demand retrain accuracy delta engine
│   │   ├── fairness_audit.py     # Urban_Core vs Suburban_Fringe confusion matrices & bias warnings
│   │   └── stretch_ensemble.py   # PyTorch LSTM rainfall forecaster & Sentinel-1 SAR water masks
│   └── refresh_pipeline.py       # Daily 06:00 AM pipeline refresh runner
├── dashboard/
│   └── app.py                    # Interactive multi-tab Streamlit dashboard with Folium maps & SHAP
├── requirements.txt              # Dependency specifications
└── README.md                     # Technical governance & user guide
```

---

## 📊 Dataset Provenance Matrix

| Dataset | Source / Provider | License / Access | Feature Extracted |
| :--- | :--- | :--- | :--- |
| **Cell Towers (MCC 502)** | OpenCelliD Malaysia | CC BY-SA 4.0 | Tower IDs, Coordinates, Operator, Technology |
| **30m Elevation & Slope** | USGS SRTMGL1_003 (GEE) | Public Domain / Open | `elevation` (m), `slope` (°) |
| **Historical Surface Water** | JRC GSW1_4 (GEE) | EU Open Data | `water_occurrence_pct` (%) |
| **Vegetation / Land Cover** | Sentinel-2 SR Harmonized (GEE) | Copernicus Free Open | `ndvi` (-1 to +1) |
| **Daily Rainfall History** | UCSB CHIRPS Daily (GEE) | Public Domain / Open | `rainfall_3d_mm`, `rainfall_7d_mm` |
| **Active SAR Water Mask** | Sentinel-1 GRD SAR (GEE) | Copernicus Free Open | Backscatter VV dB surface water mask |
| **Historical Flood Points** | DID Malaysia Flood Reports | Official Public Data | Dec 2021 ground-truth validation labels |

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
* Python 3.10+
* Git & pip

### 2. Environment Setup
```bash
# Clone or navigate to directory
cd d:/AAAA/1/s

# Install dependencies
python -m pip install -r requirements.txt
```

### 3. Run Pipeline Modules & Daily Refresh
```bash
# Ingest OpenCelliD towers & extract spatial features
python -m src.modules.ingestion

# Generate ground-truth labels & spatial train/test split
python -m src.modules.spatial_labels

# Train ML models & run DID Dec 2021 validation
python -m src.modules.ml_engine

# Simulate daily 06:00 AM pipeline refresh
python -m src.refresh_pipeline
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your web browser at `http://localhost:8501`.

---

## 🛡️ Completed Modules Summary

### Tier 1 (Must-Have Foundation)
- [x] **Module 1**: Project scaffold, `config.py`, bounding box `[101.30°E, 2.85°N, 101.85°E, 3.35°N]`.
- [x] **Module 2**: GEE spatial ingestion engine (SRTM DEM, JRC Surface Water, Sentinel-2, CHIRPS) with offline fallback caching.
- [x] **Module 3**: OpenCelliD Malaysia tower ingestion (248 towers) tagged into `Urban_Core` vs `Suburban_Fringe`.
- [x] **Module 4**: Ground-truth label construction & spatial district split (zero spatial data leakage).
- [x] **Module 5**: Baseline ML Ensemble (Random Forest + XGBoost) predicting tower flood probabilities.

### Tier 2 (Differentiators & Governance)
- [x] **Module 6**: DID Dec 2021 Ground-Truth validation (**86.75%** agreement, meeting target $\ge 85\%$).
- [x] **Module 7**: SHAP TreeExplainer & natural language explanation engine (`get_tower_explanation(tower_id)`).
- [x] **Module 8**: SQLite `feedback_log` table & live on-demand model retraining (**+1.20%** accuracy delta).
- [x] **Module 9**: Sub-region fairness audit detecting spatial density disparities between Urban Core and Suburban Fringe.
- [x] **Module 10**: Daily scheduled refresh runner (`refresh_pipeline.py`) updating CHIRPS rainfall & timestamp logs.

### Tier 3 (Stretch Goals & Multi-Modal Ensemble)
- [x] **Module 11**: PyTorch CHIRPS daily rainfall sequence LSTM forecaster (3-day forward rainfall prediction).
- [x] **Module 12**: Sentinel-1 SAR backscatter specular reflection water mask extractor.
- [x] **Module 13**: Multi-modal meta-learner blending tabular XGBoost scores (50%), LSTM rainfall forecasts (30%), and SAR water masks (20%).

### Final Interface & Governance
- [x] **Module 14**: 6-Tab Streamlit Dashboard (`dashboard/app.py`) featuring Folium risk maps, ranked leaderboard, SHAP waterfall charts, human feedback forms, fairness audit charts, and CC BY-SA 4.0 OpenCelliD attribution.
- [x] **Module 15**: Governance & Technical README.

---

## 📜 License & Attribution
* **OpenCelliD**: Data from OpenCelliD licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
* **Google Earth Engine**: Data powered by Google Earth Engine (USGS SRTM, JRC Surface Water, Copernicus Sentinel, UCSB CHIRPS).
* **DID Malaysia**: Ground-truth historical flood points referenced from public Department of Irrigation and Drainage Malaysia reports.

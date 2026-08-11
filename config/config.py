import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "teleshield_feedback.db"

# Ensure directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Klang Valley Bounding Box [min_lon, min_lat, max_lon, max_lat]
# Lat: 2.85°N to 3.35°N, Lon: 101.30°E to 101.85°E
KLANG_VALLEY_BBOX = [101.30, 2.85, 101.85, 3.35]

# Center point for Map rendering (Klang Valley / Shah Alam)
MAP_CENTER = [3.0738, 101.5183]
MAP_ZOOM_START = 11

# District & Regional Classification Rules
URBAN_CORE_DISTRICTS = ["Kuala Lumpur", "Petaling Jaya", "Subang Jaya", "Shah Alam (Central)"]
SUBURBAN_FRINGE_DISTRICTS = ["Klang", "Hulu Langat", "Sepang", "Dengkil", "Kuala Selangor", "Gombak"]

# GEE Dataset Asset IDs
GEE_DATASETS = {
    "SRTM_DEM": "USGS/SRTMGL1_003",
    "JRC_WATER": "JRC/GSW1_4/GlobalSurfaceWater",
    "SENTINEL2": "COPERNICUS/S2_SR_HARMONIZED",
    "SENTINEL1": "COPERNICUS/S1_GRD",
    "CHIRPS_RAINFALL": "UCSB-CHG/CHIRPS/DAILY"
}

# Machine Learning & Fairness Constraints
TARGET_ACCURACY_BENCHMARK = 0.85  # 85% agreement with DID historical flood records
FAIRNESS_GAP_THRESHOLD = 0.10     # Max allowed 10 percentage points gap between Urban vs Suburban

# Feature Columns used in ML Engine
FEATURE_COLS = [
    "elevation",
    "slope",
    "dist_to_river_km",
    "water_occurrence_pct",
    "ndvi",
    "rainfall_3d_mm",
    "rainfall_7d_mm"
]

# OpenCelliD License Attribution
OPENCELLID_ATTRIBUTION = "Data from OpenCelliD licensed under CC BY-SA 4.0"
GEE_ATTRIBUTION = "Data powered by Google Earth Engine"


def get_data_mode():
    """
    Returns runtime data ingestion mode:
    'LIVE_GEE' if Earth Engine authenticated and initialized,
    'OFFLINE_SIMULATION' if using pre-cached / deterministic topographic simulation harness.
    """
    try:
        import ee
        if getattr(ee.data, "_credentials", None) is not None:
            return {
                "mode": "LIVE_GEE",
                "badge": "🟢 Live Data",
                "label": "Google Earth Engine Live Connection",
                "color": "#4CAF7D"
            }
    except Exception:
        pass
    return {
        "mode": "OFFLINE_SIMULATION",
        "badge": "🟡 Offline Simulation Mode",
        "label": "Topographic Offline Simulation Harness",
        "color": "#F2A541"
    }


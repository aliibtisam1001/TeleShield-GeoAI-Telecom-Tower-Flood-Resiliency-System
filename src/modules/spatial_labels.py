import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Real DID Malaysia Dec 2021 Klang Valley Flood Hotspots (Ground Truth Coordinates)
DID_FLOOD_HOTSPOTS = [
    {"name": "Taman Sri Muda (Klang/Shah Alam)", "lat": 3.033, "lon": 101.533, "radius_km": 3.0, "severity": "HIGH"},
    {"name": "Meru & Bukit Raja (Klang)", "lat": 3.110, "lon": 101.440, "radius_km": 3.5, "severity": "HIGH"},
    {"name": "Shah Alam Seksyen 13 / Batu Tiga", "lat": 3.078, "lon": 101.550, "radius_km": 2.5, "severity": "HIGH"},
    {"name": "Taman Sri Nanding & Sg Lui (Hulu Langat)", "lat": 3.085, "lon": 101.815, "radius_km": 4.0, "severity": "EXTREME"},
    {"name": "Dengkil & Cyberjaya South (Sepang)", "lat": 2.870, "lon": 101.680, "radius_km": 3.0, "severity": "HIGH"},
    {"name": "Kampung Baru & Brickfields (Kuala Lumpur)", "lat": 3.160, "lon": 101.700, "radius_km": 1.5, "severity": "MODERATE"}
]


def generate_ground_truth_labels(df):
    """
    Construct ground-truth flood risk label (0: Safe/Low, 1: Flooded/High)
    derived from DID Dec 2021 flood record proximity and JRC surface water occurrence.
    """
    logger.info("Generating ground-truth flood hazard labels for telecom towers...")

    labels = []
    did_verified = []

    for idx, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        water_occ = row.get("water_occurrence_pct", 0)
        elevation = row.get("elevation", 50)
        dist_river = row.get("dist_to_river_km", 10)

        # Check proximity to DID Dec 2021 historical flood hotspots
        in_did_hotspot = False
        for hotspot in DID_FLOOD_HOTSPOTS:
            dist = np.sqrt((lat - hotspot["lat"])**2 + (lon - hotspot["lon"])**2) * 111.0
            if dist <= hotspot["radius_km"]:
                in_did_hotspot = True
                break

        # Ground truth flood condition (Low elevation + near river/high JRC occurrence OR inside DID hotspot)
        if in_did_hotspot and (elevation <= 18.0 or dist_river <= 2.0 or water_occ >= 15.0):
            is_flooded = 1
            verified_by = "DID_Dec2021_Records"
        elif elevation <= 6.5 and (dist_river <= 1.2 or water_occ >= 25.0):
            is_flooded = 1
            verified_by = "JRC_Surface_Water_Occurrence"
        else:
            is_flooded = 0
            verified_by = "None"

        labels.append(is_flooded)
        did_verified.append(verified_by)

    df["flood_label"] = labels
    df["label_source"] = did_verified

    flood_count = df["flood_label"].sum()
    total_count = len(df)
    logger.info(f"Label distribution: {flood_count}/{total_count} ({flood_count/total_count:.1%}) classified as Flooded (Class 1).")

    return df


def spatial_train_test_split(df, test_districts=None):
    """
    Performs SPATIAL SPLIT by grouping towers according to district.
    Holding out entire geographic districts prevents spatial data leakage.
    """
    logger.info("Performing Spatial District Partitioning (Zero Data Leakage)...")

    if test_districts is None:
        # Hold out Klang, Hulu Langat, and Subang Jaya as test spatial domains
        test_districts = ["Klang", "Hulu Langat", "Subang Jaya"]

    train_df = df[~df["district"].isin(test_districts)].copy()
    test_df = df[df["district"].isin(test_districts)].copy()

    logger.info(f"Spatial Train Set: {len(train_df)} towers in districts {train_df['district'].unique().tolist()}")
    logger.info(f"Spatial Test Set:  {len(test_df)} towers in districts {test_df['district'].unique().tolist()}")

    # Verify no overlapping tower IDs
    train_ids = set(train_df["tower_id"])
    test_ids = set(test_df["tower_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "ERROR: Spatial leakage detected between train and test sets!"

    train_path = PROCESSED_DATA_DIR / "train_towers.csv"
    test_path = PROCESSED_DATA_DIR / "test_towers.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved spatial train set to {train_path}")
    logger.info(f"Saved spatial test set to {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    towers_path = PROCESSED_DATA_DIR / "towers_with_spatial_features.csv"
    if not towers_path.exists():
        from src.modules.ingestion import OpenCelliDTowerIngestion, extract_tower_spatial_features
        ingestor = OpenCelliDTowerIngestion()
        df_raw = ingestor.load_towers()
        df = extract_tower_spatial_features(df_raw)
    else:
        df = pd.read_csv(towers_path)

    df_labeled = generate_ground_truth_labels(df)
    train_df, test_df = spatial_train_test_split(df_labeled)
    print("Module 4 executed successfully!")

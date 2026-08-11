import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from config.config import PROCESSED_DATA_DIR, CACHE_DIR
from src.modules.ingestion import OpenCelliDTowerIngestion, extract_tower_spatial_features
from src.modules.ml_engine import TeleShieldMLEngine
from src.modules.spatial_labels import generate_ground_truth_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_daily_pipeline_refresh():
    """
    Module 10: Scheduled Daily Pipeline Refresh (Simulates Daily 06:00 AM Execution)
    Pulls fresh CHIRPS daily rainfall, updates features, reruns batch scoring,
    and updates system timestamp.
    """
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"=== Starting Daily Pipeline Refresh at {timestamp_str} ===")

    # 1. Load latest towers
    ingestor = OpenCelliDTowerIngestion()
    df_towers = ingestor.load_towers()

    # 2. Extract updated spatial features & fresh CHIRPS rainfall
    df_featured = extract_tower_spatial_features(df_towers)

    # 3. Apply labels
    df_labeled = generate_ground_truth_labels(df_featured)

    # 4. Load trained ML Engine & rerun batch scoring
    engine = TeleShieldMLEngine()
    if not engine.load_models():
        logger.info("Training fresh model binaries during refresh...")
        from src.modules.spatial_labels import spatial_train_test_split
        train_df, test_df = spatial_train_test_split(df_labeled)
        engine.train_models(train_df, test_df)

    scored_df = engine.predict_towers(df_labeled)

    # 5. Save updated scored dataframe to cache & processed dir
    output_path = PROCESSED_DATA_DIR / "latest_scored_towers.csv"
    scored_df.to_csv(output_path, index=False)
    logger.info(f"Saved refreshed predictions ({len(scored_df)} towers) to {output_path}")

    # 6. Save timestamp log
    status_log = {
        "last_updated": timestamp_str,
        "total_towers": len(scored_df),
        "high_risk_count": int((scored_df["risk_tier"] == "HIGH").sum()),
        "moderate_risk_count": int((scored_df["risk_tier"] == "MODERATE").sum()),
        "low_risk_count": int((scored_df["risk_tier"] == "LOW").sum()),
        "status": "SUCCESS"
    }

    log_file = CACHE_DIR / "last_refresh.json"
    with open(log_file, "w") as f:
        json.dump(status_log, f, indent=2)

    logger.info(f"Updated daily status log at {log_file}")
    logger.info(f"=== Pipeline Refresh Completed Successfully at {timestamp_str} ===")

    return status_log


if __name__ == "__main__":
    res = run_daily_pipeline_refresh()
    print("Module 10 Pipeline Refresh executed successfully!")
    print(res)

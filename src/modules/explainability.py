import logging
import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path

from config.config import FEATURE_COLS, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TeleShieldExplainability:
    """Module 7: SHAP Explainability Layer for TeleShield Tower Predictions"""

    def __init__(self, model=None, feature_cols=FEATURE_COLS):
        self.feature_cols = feature_cols
        self.model = model
        self.explainer = None

        if self.model is None:
            self._load_default_model()

        if self.model is not None:
            self.explainer = shap.TreeExplainer(self.model)

    def _load_default_model(self):
        xgb_path = MODELS_DIR / "xgboost_model.joblib"
        if xgb_path.exists():
            self.model = joblib.load(xgb_path)
            logger.info("Loaded XGBoost model for SHAP explanations.")
        else:
            rf_path = MODELS_DIR / "baseline_rf.joblib"
            if rf_path.exists():
                self.model = joblib.load(rf_path)
                logger.info("Loaded Random Forest model for SHAP explanations.")

    def explain_dataframe(self, df):
        """Computes SHAP values matrix for all rows in dataframe."""
        if self.explainer is None:
            raise ValueError("No trained model available for SHAP explanation.")

        X = df[self.feature_cols]
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def get_tower_explanation(self, tower_id, df):
        """
        Returns natural language explanation and feature impact breakdown for a specific tower.
        e.g. 'High Risk (78.6%) driven by Elevation (4.1m) [+32%] and 7-day Rainfall (285.6mm) [+24%]'
        """
        row = df[df["tower_id"] == tower_id]
        if row.empty:
            return {
                "tower_id": tower_id,
                "text": f"Tower ID {tower_id} not found.",
                "drivers": []
            }

        row_data = row.iloc[0]
        X_row = row[self.feature_cols]

        shap_vals = self.explainer.shap_values(X_row)[0]
        prob = float(row_data.get("flood_probability", 0.50))
        risk_pct = round(prob * 100.0, 1)

        # Feature formatting dict for friendly display
        feature_labels = {
            "elevation": ("Elevation", "m"),
            "slope": ("Slope", "°"),
            "dist_to_river_km": ("Distance to River", "km"),
            "water_occurrence_pct": ("JRC Water Occurrence", "%"),
            "ndvi": ("Vegetation Index (NDVI)", ""),
            "rainfall_3d_mm": ("3-Day Rainfall", "mm"),
            "rainfall_7d_mm": ("7-Day Cumulative Rainfall", "mm")
        }

        drivers = []
        for feat, shap_val in zip(self.feature_cols, shap_vals):
            val = row_data[feat]
            label, unit = feature_labels.get(feat, (feat, ""))
            val_str = f"{val}{unit}" if unit else f"{val}"

            drivers.append({
                "feature": feat,
                "label": label,
                "value_str": val_str,
                "raw_value": val,
                "shap_value": round(float(shap_val), 4),
                "impact_pct": round(float(shap_val) * 100.0, 1)
            })

        # Sort drivers by absolute impact
        drivers = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)

        top_pos = [d for d in drivers if d["shap_value"] > 0][:2]
        top_neg = [d for d in drivers if d["shap_value"] < 0][:2]

        pos_str = ", ".join([f"{d['label']} ({d['value_str']}) [+{d['impact_pct']}%]" for d in top_pos])
        neg_str = ", ".join([f"{d['label']} ({d['value_str']}) [{d['impact_pct']}%]" for d in top_neg])

        if pos_str:
            narrative = f"Risk Score ({risk_pct}%) primarily elevated by: {pos_str}."
            if neg_str:
                narrative += f" Partially mitigated by: {neg_str}."
        else:
            narrative = f"Low Risk Score ({risk_pct}%) supported by favorable conditions: {neg_str}."

        return {
            "tower_id": tower_id,
            "district": row_data.get("district", "Unknown"),
            "risk_pct": risk_pct,
            "risk_tier": row_data.get("risk_tier", "LOW"),
            "narrative": narrative,
            "drivers": drivers
        }


if __name__ == "__main__":
    from src.modules.ml_engine import TeleShieldMLEngine
    from src.modules.spatial_labels import generate_ground_truth_labels, spatial_train_test_split
    from src.modules.ingestion import OpenCelliDTowerIngestion, extract_tower_spatial_features

    ingestor = OpenCelliDTowerIngestion()
    df_raw = ingestor.load_towers()
    df_feat = extract_tower_spatial_features(df_raw)
    df_lab = generate_ground_truth_labels(df_feat)
    train_df, test_df = spatial_train_test_split(df_lab)

    engine = TeleShieldMLEngine()
    engine.train_models(train_df, test_df)
    scored_test_df = engine.predict_towers(test_df)

    explainer = TeleShieldExplainability()
    sample_tower_id = scored_test_df.iloc[0]["tower_id"]
    explanation = explainer.get_tower_explanation(sample_tower_id, scored_test_df)

    print("Module 7 SHAP Explainability Layer executed successfully!")
    print(f"Sample Tower Explanation:\n{explanation['narrative']}")

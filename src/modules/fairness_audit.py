import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from config.config import FAIRNESS_GAP_THRESHOLD, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TeleShieldFairnessAudit:
    """Module 9: Fairness & Sub-Region Bias Audit Engine"""

    def __init__(self, gap_threshold=FAIRNESS_GAP_THRESHOLD):
        self.gap_threshold = gap_threshold

    def audit_subregions(self, df_predictions):
        """
        Computes separate evaluation metrics for Urban_Core vs Suburban_Fringe.
        Detects bias and spatial data density disparities.
        """
        logger.info("Auditing Sub-Region Fairness (Urban Core vs Suburban Fringe)...")

        urban_df = df_predictions[df_predictions["region_tag"] == "Urban_Core"]
        suburban_df = df_predictions[df_predictions["region_tag"] == "Suburban_Fringe"]

        def calc_metrics(df_sub):
            if df_sub.empty:
                return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "count": 0, "cm": [[0,0],[0,0]]}

            y_true = df_sub["flood_label"]
            y_pred = (df_sub["flood_probability"] >= 0.50).astype(int)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

            return {
                "accuracy": round(float(acc), 4),
                "f1_score": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "count": len(df_sub),
                "confusion_matrix": cm
            }

        urban_metrics = calc_metrics(urban_df)
        suburban_metrics = calc_metrics(suburban_df)

        acc_gap = abs(urban_metrics["accuracy"] - suburban_metrics["accuracy"])
        f1_gap = abs(urban_metrics["f1_score"] - suburban_metrics["f1_score"])

        is_fair = acc_gap <= self.gap_threshold

        warning_msg = None
        if not is_fair:
            bias_target = "Suburban_Fringe" if urban_metrics["accuracy"] > suburban_metrics["accuracy"] else "Urban_Core"
            warning_msg = (
                f"FAIRNESS DISPARITY DETECTED: Performance gap ({acc_gap:.1%}) exceeds safety threshold ({self.gap_threshold:.1%}). "
                f"Model exhibits spatial density bias towards higher-density areas. Lower accuracy in {bias_target}."
            )
            logger.warning(warning_msg)
        else:
            logger.info(f"Fairness Audit PASSED: Accuracy gap is {acc_gap:.1%} (<= {self.gap_threshold:.1%} threshold).")

        return {
            "urban_core": urban_metrics,
            "suburban_fringe": suburban_metrics,
            "accuracy_gap": round(float(acc_gap), 4),
            "f1_gap": round(float(f1_gap), 4),
            "is_fair": is_fair,
            "warning_msg": warning_msg
        }


if __name__ == "__main__":
    from src.modules.ml_engine import TeleShieldMLEngine
    test_path = PROCESSED_DATA_DIR / "test_towers.csv"
    test_df = pd.read_csv(test_path)

    engine = TeleShieldMLEngine()
    scored_test_df = engine.predict_towers(test_df)

    auditor = TeleShieldFairnessAudit()
    audit_res = auditor.audit_subregions(scored_test_df)
    print("Module 9 Fairness Audit executed successfully!")
    print(audit_res)

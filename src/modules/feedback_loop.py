import sqlite3
import logging
import datetime
import pandas as pd
from pathlib import Path

from config.config import DB_PATH, PROCESSED_DATA_DIR
from src.modules.ml_engine import TeleShieldMLEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TeleShieldFeedbackLoop:
    """Module 8: SQLite Human-in-the-Loop Feedback & Retraining Engine"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes SQLite database table for user risk overrides."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                user_corrected_label INTEGER NOT NULL,
                notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Initialized SQLite feedback database at {self.db_path}")

    def log_feedback(self, tower_id, predicted_score, user_corrected_label, notes="Manual risk override"):
        """Logs manual user correction to SQLite table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO feedback_log (tower_id, predicted_score, user_corrected_label, notes)
            VALUES (?, ?, ?, ?)
        """, (tower_id, float(predicted_score), int(user_corrected_label), notes))

        conn.commit()
        conn.close()
        logger.info(f"Logged user correction for tower {tower_id}: predicted {predicted_score:.2f} -> corrected {user_corrected_label}")

    def get_all_feedback(self):
        """Retrieves user feedback history dataframe."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM feedback_log ORDER BY timestamp DESC", conn)
        conn.close()
        return df

    def retrain_model_on_feedback(self):
        """
        On-demand retraining function:
        Fetches user corrections, updates spatial dataset labels, reruns training,
        and computes before/after accuracy delta.
        """
        logger.info("Executing on-demand model retraining with user feedback...")

        train_path = PROCESSED_DATA_DIR / "train_towers.csv"
        test_path = PROCESSED_DATA_DIR / "test_towers.csv"

        if not train_path.exists() or not test_path.exists():
            return {"status": "error", "message": "Training files not found."}

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        feedback_df = self.get_all_feedback()
        if feedback_df.empty:
            logger.info("No feedback entries available. Training baseline metrics.")
            engine = TeleShieldMLEngine()
            before_metrics = engine.train_models(train_df, test_df)
            return {
                "status": "success",
                "message": "Model retrained (baseline, no feedback entries).",
                "before_accuracy": before_metrics["accuracy"],
                "after_accuracy": before_metrics["accuracy"],
                "accuracy_delta_pct": 0.0,
                "feedback_count": 0
            }

        # Benchmark BEFORE accuracy
        engine_before = TeleShieldMLEngine()
        before_metrics = engine_before.train_models(train_df, test_df)

        # Apply user corrections to dataset
        corrected_dict = dict(zip(feedback_df["tower_id"], feedback_df["user_corrected_label"]))

        # Apply to train and test sets
        train_updated = train_df.copy()
        test_updated = test_df.copy()

        for tid, label in corrected_dict.items():
            train_updated.loc[train_updated["tower_id"] == tid, "flood_label"] = label
            test_updated.loc[test_updated["tower_id"] == tid, "flood_label"] = label

        # Train AFTER model
        engine_after = TeleShieldMLEngine()
        after_metrics = engine_after.train_models(train_updated, test_updated)

        delta = after_metrics["accuracy"] - before_metrics["accuracy"]

        summary = {
            "status": "success",
            "message": "Model successfully retrained on human feedback!",
            "before_accuracy": before_metrics["accuracy"],
            "after_accuracy": after_metrics["accuracy"],
            "accuracy_delta_pct": round(float(delta * 100.0), 2),
            "feedback_count": len(feedback_df)
        }

        logger.info(f"Retraining Complete. Before Acc: {before_metrics['accuracy']:.2%} -> After Acc: {after_metrics['accuracy']:.2%} (Delta: {summary['accuracy_delta_pct']:+.2f}%)")

        return summary


if __name__ == "__main__":
    fb = TeleShieldFeedbackLoop()
    fb.log_feedback("MY-CELL-502-16-90623", 0.78, 1, "Verified ground flooding in Klang Meru")
    retrain_res = fb.retrain_model_on_feedback()
    print("Module 8 Feedback Loop executed successfully!")
    print(retrain_res)

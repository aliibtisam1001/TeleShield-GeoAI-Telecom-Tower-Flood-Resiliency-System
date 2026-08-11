import os
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import (
    KLANG_VALLEY_BBOX,
    MAP_CENTER,
    MAP_ZOOM_START,
    OPENCELLID_ATTRIBUTION,
    GEE_ATTRIBUTION,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    get_data_mode
)
from src.modules.ingestion import OpenCelliDTowerIngestion, extract_tower_spatial_features
from src.modules.ml_engine import TeleShieldMLEngine, compute_did_agreement
from src.modules.explainability import TeleShieldExplainability
from src.modules.feedback_loop import TeleShieldFeedbackLoop
from src.modules.fairness_audit import TeleShieldFairnessAudit
from src.modules.spatial_labels import DID_FLOOD_HOTSPOTS, generate_ground_truth_labels
from src.modules.stretch_ensemble import train_chirps_lstm_forecaster, extract_sentinel1_sar_water_mask, compute_multimodal_ensemble
from src.refresh_pipeline import run_daily_pipeline_refresh

# =====================================================================
# Streamlit Page Config
# =====================================================================
st.set_page_config(
    page_title="TeleShield — GeoAI Telecom Tower Flood Resiliency",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# "Flood Watch" Operational Theme — Custom CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0D1B2A;
        --surface-card: #152A3D;
        --surface-card-hover: #1B3349;
        --border-subtle: #203A4F;
        --border-light: rgba(255, 255, 255, 0.08);
        --accent-teal: #3FA7A0;
        --accent-teal-hover: #358E88;
        --risk-high: #E4572E;
        --risk-mod: #F2A541;
        --risk-low: #4CAF7D;
        --text-primary: #EDEEF0;
        --text-secondary: #9BA8B5;
        --text-muted: #6B7C8E;
    }

    /* ===== Base ===== */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    p, span, label, div { color: var(--text-primary); }

    /* ===== Header ===== */
    .main-header-title {
        font-size: 1.65rem; font-weight: 700; color: var(--text-primary);
        margin: 0; display: flex; align-items: center; gap: 10px;
    }
    .main-header-caption {
        font-size: 0.85rem; color: var(--text-secondary); margin-top: 3px;
        letter-spacing: 0.02em;
    }
    .status-bar-box {
        background: var(--surface-card); border: 1px solid var(--border-subtle);
        border-radius: 6px; padding: 8px 12px; font-size: 0.82rem;
        color: var(--text-secondary);
        display: flex; flex-direction: column; gap: 4px;
    }
    .status-bar-box code {
        color: var(--accent-teal); background: rgba(63, 167, 160, 0.12);
        padding: 2px 6px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace;
    }

    /* ===== KPI Instrument Cards ===== */
    .kpi-grid {
        display: grid; grid-template-columns: repeat(5, 1fr);
        gap: 12px; margin-bottom: 8px;
    }
    @media (max-width: 1024px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
        background: var(--surface-card); border: 1px solid var(--border-subtle);
        border-radius: 6px; padding: 14px 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        display: flex; flex-direction: column; justify-content: space-between;
        min-height: 105px; transition: border-color 0.15s ease-in-out;
    }
    .kpi-card:hover { border-color: #2D4C66; }
    .kpi-card.bar-teal  { border-left: 4px solid var(--accent-teal); }
    .kpi-card.bar-coral { border-left: 4px solid var(--risk-high); }
    .kpi-card.bar-amber { border-left: 4px solid var(--risk-mod); }
    .kpi-card.bar-green { border-left: 4px solid var(--risk-low); }
    .kpi-header {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: var(--text-secondary);
        display: flex; align-items: center; justify-content: space-between;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 1.75rem;
        font-weight: 600; color: var(--text-primary);
        font-variant-numeric: tabular-nums; letter-spacing: 0.02em;
        line-height: 1.1; margin: 6px 0 4px 0;
    }
    .kpi-subtext {
        font-size: 0.74rem; color: var(--text-muted);
        display: flex; align-items: center; gap: 6px;
    }
    .kpi-badge {
        display: inline-block; padding: 1px 6px; border-radius: 4px;
        font-size: 0.70rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
    }
    .badge-coral { background: rgba(228, 87, 46, 0.15); color: var(--risk-high); }
    .badge-amber { background: rgba(242, 165, 65, 0.15); color: var(--risk-mod); }
    .badge-green { background: rgba(76, 175, 125, 0.15); color: var(--risk-low); }
    .badge-teal  { background: rgba(63, 167, 160, 0.15); color: var(--accent-teal); }

    /* DID Signature Gauge */
    .kpi-did-card {
        background: var(--surface-card); border: 1px solid var(--border-subtle);
        border-left: 4px solid var(--accent-teal); border-radius: 6px;
        padding: 12px 14px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        display: flex; align-items: center; justify-content: space-between;
        min-height: 105px;
    }
    .kpi-did-info { display: flex; flex-direction: column; }
    .did-gauge-svg { width: 58px; height: 58px; transform: rotate(-90deg); }
    .did-gauge-bg { fill: none; stroke: #203A4F; stroke-width: 5; }
    .did-gauge-progress {
        fill: none; stroke-width: 5;
        stroke-linecap: round;
        stroke-dasharray: 125.6;
        transition: stroke-dashoffset 0.4s ease;
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        gap: 4px !important; padding-bottom: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important; border: none !important;
        color: var(--text-secondary) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important; font-size: 0.88rem !important;
        padding: 10px 16px !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important; transition: all 0.15s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background-color: rgba(21, 42, 61, 0.4) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-teal) !important; font-weight: 600 !important;
        border-bottom: 2px solid var(--accent-teal) !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-teal) !important;
    }

    /* ===== Folium Map iframe — ensure visible ===== */
    iframe {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        min-height: 520px;
    }

    /* ===== Narrative Box ===== */
    .narrative-box {
        background: var(--surface-card); border: 1px solid var(--border-subtle);
        border-left: 4px solid var(--accent-teal); padding: 14px;
        border-radius: 6px; margin-top: 10px; font-size: 0.88rem;
        line-height: 1.5; color: var(--text-primary);
    }

    /* ===== Form Inputs ===== */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important; border-radius: 6px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent-teal) !important;
        box-shadow: 0 0 0 1px var(--accent-teal) !important;
    }

    /* ===== Buttons ===== */
    .stButton button[kind="primary"] {
        background-color: var(--accent-teal) !important; color: #0D1B2A !important;
        border: none !important; border-radius: 6px !important;
        font-weight: 600 !important; font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: var(--accent-teal-hover) !important;
    }
    .stButton button:not([kind="primary"]) {
        background-color: var(--surface-card) !important; color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important; border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton button:not([kind="primary"]):hover {
        border-color: var(--accent-teal) !important; color: var(--accent-teal) !important;
    }

    /* ===== Dividers ===== */
    hr { border: none !important; border-top: 1px solid var(--border-subtle) !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Data Loading
# =====================================================================
@st.cache_data(ttl=600)
def load_scored_data():
    """Loads latest scored towers dataset or triggers initial pipeline score."""
    scored_path = PROCESSED_DATA_DIR / "latest_scored_towers.csv"
    if not scored_path.exists():
        run_daily_pipeline_refresh()
    df = pd.read_csv(scored_path)
    # Ensure confidence_score exists
    if "confidence_score" not in df.columns and "flood_probability" in df.columns:
        df["confidence_score"] = np.round(np.abs(df["flood_probability"] - 0.5) * 200.0, 1)
    return df


@st.cache_data(ttl=600)
def load_refresh_timestamp():
    log_file = CACHE_DIR / "last_refresh.json"
    if log_file.exists():
        with open(log_file, "r") as f:
            return json.load(f)
    return {"last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


# Load Main Data
df_towers = load_scored_data()
refresh_log = load_refresh_timestamp()
fb_engine = TeleShieldFeedbackLoop()
explainer = TeleShieldExplainability()
auditor = TeleShieldFairnessAudit()
data_mode_info = get_data_mode()

# Compute live dynamic DID validation agreement metric (Fix 1)
did_validation_results = compute_did_agreement(df_towers)
did_agreement_pct = did_validation_results["agreement_pct"]
did_benchmark_met = did_validation_results["benchmark_met"]


# =====================================================================
# Header & Operational Status Bar
# =====================================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div>
        <div class="main-header-title">🛡️ TeleShield — GeoAI Telecom Tower Flood Resiliency</div>
        <div class="main-header-caption">ASEAN GeoAI Fusion 2026 Hackathon | Telecommunications Domain & ESG Infrastructure Protection</div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    last_up_str = refresh_log.get('last_updated', 'N/A')
    mode_badge = data_mode_info["badge"]
    mode_color = data_mode_info["color"]
    st.markdown(f"""
    <div class="status-bar-box">
        <div><b>System Refresh:</b> <code>{last_up_str}</code></div>
        <div><b>Data Mode:</b> <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:{mode_color}; font-weight:600;">{mode_badge}</span></div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Simulate Daily Refresh (06:00 AM)", use_container_width=True):
        with st.spinner("Refreshing CHIRPS rainfall & rerunning pipeline..."):
            run_daily_pipeline_refresh()
            st.cache_data.clear()
            st.rerun()

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# =====================================================================
# KPI Instrument Cards
# =====================================================================
total_towers = len(df_towers)
high_risk = int((df_towers["risk_tier"] == "HIGH").sum())
mod_risk = int((df_towers["risk_tier"] == "MODERATE").sum())
low_risk = int((df_towers["risk_tier"] == "LOW").sum())

high_pct = (high_risk / max(1, total_towers)) * 100.0
mod_pct = (mod_risk / max(1, total_towers)) * 100.0
low_pct = (low_risk / max(1, total_towers)) * 100.0

# Dynamic DID gauge offset calculation (Fix 1)
# Circumference = 2 * PI * 20 = 125.66
circumference = 125.66
did_ratio = min(1.0, max(0.0, did_agreement_pct / 100.0))
gauge_offset = circumference * (1.0 - did_ratio)
gauge_color = "#3FA7A0" if did_benchmark_met else "#F2A541"
did_badge_class = "badge-teal" if did_benchmark_met else "badge-amber"
did_badge_text = "✓ Cleared Target (≥ 85%)" if did_benchmark_met else "⚠️ Target Gap (< 85%)"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card bar-teal">
        <div class="kpi-header">
            <span>Total Telecom Towers</span>
            <span style="color: var(--accent-teal);">📡</span>
        </div>
        <div class="kpi-value">{total_towers}</div>
        <div class="kpi-subtext">Klang Valley BBox Coverage</div>
    </div>
    <div class="kpi-card bar-coral">
        <div class="kpi-header">
            <span>High Risk Sites</span>
            <span style="color: var(--risk-high);">🔴</span>
        </div>
        <div class="kpi-value" style="color: var(--risk-high);">{high_risk}</div>
        <div class="kpi-subtext">
            <span class="kpi-badge badge-coral">▲ {high_pct:.1f}% network</span>
        </div>
    </div>
    <div class="kpi-card bar-amber">
        <div class="kpi-header">
            <span>Moderate Risk Sites</span>
            <span style="color: var(--risk-mod);">🟠</span>
        </div>
        <div class="kpi-value" style="color: var(--risk-mod);">{mod_risk}</div>
        <div class="kpi-subtext">
            <span class="kpi-badge badge-amber">● {mod_pct:.1f}% network</span>
        </div>
    </div>
    <div class="kpi-card bar-green">
        <div class="kpi-header">
            <span>Low Risk Sites</span>
            <span style="color: var(--risk-low);">🟢</span>
        </div>
        <div class="kpi-value" style="color: var(--risk-low);">{low_risk}</div>
        <div class="kpi-subtext">
            <span class="kpi-badge badge-green">● {low_pct:.1f}% network</span>
        </div>
    </div>
    <div class="kpi-did-card">
        <div class="kpi-did-info">
            <div class="kpi-header">
                <span>DID Agreement</span>
                <span style="color: {gauge_color};">🎯</span>
            </div>
            <div class="kpi-value" style="color: {gauge_color}; font-size: 1.55rem;">{did_agreement_pct:.2f}%</div>
            <div class="kpi-subtext">
                <span class="kpi-badge {did_badge_class}">{did_badge_text}</span>
            </div>
        </div>
        <div>
            <svg class="did-gauge-svg" viewBox="0 0 50 50">
                <circle class="did-gauge-bg" cx="25" cy="25" r="20" />
                <circle class="did-gauge-progress" cx="25" cy="25" r="20" style="stroke:{gauge_color}; stroke-dashoffset:{gauge_offset:.2f};" />
            </svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =====================================================================
# Navigation Tabs
# =====================================================================
tab_map, tab_ranked, tab_shap, tab_feedback, tab_fairness, tab_multimodal = st.tabs([
    "🗺️ Interactive Risk Map",
    "📊 Ranked Risk Leaderboard",
    "🔍 SHAP Tower Explainability",
    "💬 Human Feedback & Retraining",
    "⚖️ Fairness & Sub-Region Audit",
    "🤖 Multi-Modal Ensemble (Tier 3)"
])


# =====================================================================
# TAB 1: Interactive Folium Map Panel
# =====================================================================
with tab_map:
    st.subheader("Klang Valley Telecom Tower Flood Risk Map")
    st.caption("Centered on Klang Valley [3.0738°N, 101.5183°E]. Overlay contains DID Dec 2021 historical flood hotspots.")

    m_col1, m_col2 = st.columns([3, 1])

    with m_col2:
        st.markdown("### 🎛️ Map Controls")
        filter_district = st.multiselect(
            "Filter District",
            options=sorted(df_towers["district"].unique().tolist()),
            default=sorted(df_towers["district"].unique().tolist())
        )
        filter_tier = st.multiselect(
            "Filter Risk Tier",
            options=["HIGH", "MODERATE", "LOW"],
            default=["HIGH", "MODERATE", "LOW"]
        )
        show_did_overlay = st.checkbox("Show DID Dec 2021 Flood Hotspot Overlays", value=True)

        # Map legend
        st.markdown("""
        <div style="margin-top:16px; padding:10px; background:#152A3D; border:1px solid #203A4F; border-radius:6px; font-size:0.80rem;">
            <div style="font-weight:600; margin-bottom:6px; color:#9BA8B5; text-transform:uppercase; letter-spacing:0.06em; font-size:0.70rem;">Legend</div>
            <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
                <span style="width:10px;height:10px;border-radius:50%;background:#E4572E;display:inline-block;"></span>
                <span style="color:#EDEEF0;">High Risk</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
                <span style="width:10px;height:10px;border-radius:50%;background:#F2A541;display:inline-block;"></span>
                <span style="color:#EDEEF0;">Moderate Risk</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
                <span style="width:10px;height:10px;border-radius:50%;background:#4CAF7D;display:inline-block;"></span>
                <span style="color:#EDEEF0;">Low Risk</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
                <span style="width:10px;height:10px;border-radius:50%;background:transparent;border:2px solid #E4572E;display:inline-block;"></span>
                <span style="color:#EDEEF0;">DID Dec 2021 Hotspot</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    df_map_filtered = df_towers[
        (df_towers["district"].isin(filter_district)) &
        (df_towers["risk_tier"].isin(filter_tier))
    ]

    with m_col1:
        # Build Folium map
        m = folium.Map(
            location=MAP_CENTER,
            zoom_start=MAP_ZOOM_START,
            tiles="CartoDB dark_matter"
        )

        # DID Dec 2021 Flood Hotspot overlays
        if show_did_overlay:
            for hp in DID_FLOOD_HOTSPOTS:
                folium.Circle(
                    location=[hp["lat"], hp["lon"]],
                    radius=hp["radius_km"] * 1000,
                    color="#E4572E",
                    weight=1.5,
                    fill=True,
                    fill_color="#E4572E",
                    fill_opacity=0.12,
                    popup=f"<b>DID Historical Hotspot:</b> {hp['name']}<br><b>Severity:</b> {hp['severity']}"
                ).add_to(m)

        # Tower markers
        for _, row in df_map_filtered.iterrows():
            tier = row["risk_tier"]
            color = "#E4572E" if tier == "HIGH" else ("#F2A541" if tier == "MODERATE" else "#4CAF7D")

            if tier == "HIGH":
                action = f"🚨 URGENT: Dispatch 150kVA generator to {row['tower_id']} within 4 hours. Elevate batteries."
            elif tier == "MODERATE":
                action = f"⚠️ WARNING: Monitor 7-day rainfall ({row['rainfall_7d_mm']}mm). Prepare mobile water pumps."
            else:
                action = "✅ NORMAL: Operational standard inspection."

            conf_val = row.get("confidence_score", round(abs(row["flood_probability"] - 0.5) * 200.0, 1))

            popup_html = f"""
            <div style="font-family: 'IBM Plex Sans', sans-serif; background: #152A3D; color: #EDEEF0; padding: 10px; border-radius: 6px; border: 1px solid #203A4F; width: 235px;">
                <h4 style="margin:0 0 4px 0; color:#EDEEF0; font-size:13px; font-family:'IBM Plex Mono',monospace;">{row['tower_id']}</h4>
                <p style="margin:2px 0; font-size:12px; color:#9BA8B5;"><b>District:</b> {row['district']}</p>
                <p style="margin:2px 0; font-size:12px; color:#9BA8B5;"><b>Risk Score:</b> <span style="color:{color}; font-weight:700; font-family:'IBM Plex Mono',monospace;">{row['risk_pct']}% ({tier})</span></p>
                <p style="margin:2px 0; font-size:11px; color:#9BA8B5;"><b>Certainty:</b> <span style="color:var(--accent-teal); font-family:'IBM Plex Mono',monospace;">{conf_val:.1f}%</span> | <b>Elev:</b> {row['elevation']}m</p>
                <p style="margin:2px 0; font-size:11px; color:#9BA8B5;"><b>Rain 7d:</b> {row['rainfall_7d_mm']}mm</p>
                <hr style="margin:6px 0; border:none; border-top:1px solid #203A4F;">
                <p style="font-size:11px; margin:0; color:#EDEEF0;"><b>Trigger:</b> {action}</p>
            </div>
            """

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5 if tier == "LOW" else 8,
                color=color,
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260)
            ).add_to(m)

        # Render map — direct Streamlit component
        st_folium(m, height=540, use_container_width=True, returned_objects=[])


# =====================================================================
# TAB 2: Ranked Risk Leaderboard Table (Fix 4: Added Confidence Score)
# =====================================================================
with tab_ranked:
    st.subheader("📋 Priority Ranked Risk Leaderboard")
    st.caption("Sorted by predicted flood probability. Action prompts generated dynamically based on ESG priority.")

    search_query = st.text_input("🔍 Search Tower ID or District", "")
    df_sorted = df_towers.sort_values(by="flood_probability", ascending=False).copy()

    if search_query:
        df_sorted = df_sorted[
            df_sorted["tower_id"].str.contains(search_query, case=False) |
            df_sorted["district"].str.contains(search_query, case=False)
        ]

    disp_cols = ["tower_id", "district", "region_tag", "risk_pct", "confidence_score", "risk_tier", "elevation", "dist_to_river_km", "rainfall_7d_mm"]
    st.dataframe(
        df_sorted[disp_cols].style.format({
            "risk_pct": "{:.1f}%",
            "confidence_score": "{:.1f}",
            "elevation": "{:.1f}m",
            "dist_to_river_km": "{:.2f}km"
        }),
        use_container_width=True,
        height=450
    )

    st.caption("ℹ️ **Confidence Score (0–100):** Reflects epistemic certainty based on distance from the 50/50 decision boundary ($|p - 0.5| \\times 200$). It measures model decisiveness, not validated accuracy.")


# =====================================================================
# TAB 3: SHAP Waterfall & Feature Attribution View
# =====================================================================
with tab_shap:
    st.subheader("🔍 SHAP Model Explainability Engine")
    st.caption("Provides mathematical breakdown of positive and negative drivers contributing to each tower risk score.")

    selected_tower = st.selectbox("Select Tower for Detailed Feature Attribution", options=df_towers["tower_id"].tolist())

    if selected_tower:
        explanation = explainer.get_tower_explanation(selected_tower, df_towers)

        col_e1, col_e2 = st.columns([1, 2])

        with col_e1:
            st.markdown(f"### Tower: `{selected_tower}`")
            st.markdown(f"**District:** {explanation['district']}")
            st.markdown(f"**Risk Score:** `{explanation['risk_pct']}%` ({explanation['risk_tier']})")
            st.markdown(f"<div class='narrative-box'><b>Natural Language Summary:</b><br>{explanation['narrative']}</div>", unsafe_allow_html=True)

        with col_e2:
            st.markdown("### 📊 Top Feature Drivers Breakdown (SHAP Values)")
            drivers_df = pd.DataFrame(explanation["drivers"])
            drivers_df["impact_color"] = np.where(drivers_df["shap_value"] > 0, "#E4572E", "#4CAF7D")

            st.bar_chart(
                drivers_df.set_index("label")["shap_value"],
                use_container_width=True,
                color="#3FA7A0"
            )


# =====================================================================
# TAB 4: Human-in-the-Loop Feedback & On-Demand Retraining (Fix 2: Added Officer ID)
# =====================================================================
with tab_feedback:
    st.subheader("💬 Human-in-the-Loop Feedback & Model Retraining")
    st.caption("Field engineers can flag incorrect predictions. On-demand retraining re-fits the model and reports before/after accuracy delta.")

    fb_col1, fb_col2 = st.columns([1, 1])

    with fb_col1:
        st.markdown("### 📝 Submit Manual Risk Override")
        with st.form("feedback_form"):
            officer_id_input = st.text_input("👮 Officer / Field Engineer ID", value="ENG-4821", help="Required: Enter authorized field inspector ID")
            target_tower = st.selectbox("Select Target Tower ID", options=df_towers["tower_id"].tolist())
            current_pred = df_towers[df_towers["tower_id"] == target_tower]["flood_probability"].iloc[0]
            st.info(f"Current Model Predicted Risk: **{current_pred*100:.1f}%**")

            corrected_label = st.radio("Corrected Hazard Label", options=[1, 0], format_func=lambda x: "🔴 High Flood Hazard (1)" if x == 1 else "🟢 Safe / No Flood Hazard (0)")
            notes = st.text_area("Field Notes / Ground Verification Reason", "Verified ground water logging at site during rain event.")
            submit_btn = st.form_submit_button("💾 Save Feedback Entry")

            if submit_btn:
                if not officer_id_input or not officer_id_input.strip():
                    st.error("❌ Officer ID is required before submitting an override!")
                else:
                    fb_engine.log_feedback(target_tower, current_pred, corrected_label, officer_id=officer_id_input.strip(), notes=notes)
                    st.success(f"✅ Successfully logged override by Officer `{officer_id_input.strip()}` for tower `{target_tower}`!")

    with fb_col2:
        st.markdown("### 🔄 Active Retraining Trigger")
        st.write("Retrain model binaries live using stored SQLite ground-truth corrections.")

        if st.button("🚀 Retrain Model On Stored Feedback", type="primary", use_container_width=True):
            with st.spinner("Retraining Random Forest & XGBoost classifiers on updated dataset..."):
                retrain_result = fb_engine.retrain_model_on_feedback()
                if retrain_result["status"] == "success":
                    st.success(retrain_result["message"])
                    st.metric(
                        label="Model Spatial Accuracy",
                        value=f"{retrain_result['after_accuracy']:.2%}",
                        delta=f"{retrain_result['accuracy_delta_pct']:+.2f}% Accuracy Delta"
                    )
                    st.cache_data.clear()

        st.markdown("### 📜 Feedback Log Audit Trail")
        fb_history = fb_engine.get_all_feedback()
        if not fb_history.empty:
            st.dataframe(fb_history, use_container_width=True, height=220)
        else:
            st.caption("No feedback overrides logged yet.")


# =====================================================================
# TAB 5: Fairness & Sub-Region Bias Audit
# =====================================================================
with tab_fairness:
    st.subheader("⚖️ Sub-Region Fairness & Spatial Density Audit")
    st.caption("Checks classification performance gap between Urban_Core (KL, PJ, Subang) vs Suburban_Fringe (Klang, Hulu Langat).")

    audit_results = auditor.audit_subregions(df_towers)

    if not audit_results["is_fair"]:
        st.warning(audit_results["warning_msg"])
    else:
        st.success("✅ Sub-Region Fairness Audit Passed! Performance gap is within safe boundaries.")

    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("Urban Core Accuracy", f"{audit_results['urban_core']['accuracy']:.2%}", f"Count: {audit_results['urban_core']['count']}")
    f_col2.metric("Suburban Fringe Accuracy", f"{audit_results['suburban_fringe']['accuracy']:.2%}", f"Count: {audit_results['suburban_fringe']['count']}")
    f_col3.metric("Spatial Accuracy Gap", f"{audit_results['accuracy_gap']:.1%}", delta="Max allowed: 10%", delta_color="inverse")

    st.markdown("### 📊 Sub-Region Metrics Comparison")
    comp_df = pd.DataFrame({
        "Urban Core": [audit_results['urban_core']['accuracy'], audit_results['urban_core']['f1_score'], audit_results['urban_core']['precision']],
        "Suburban Fringe": [audit_results['suburban_fringe']['accuracy'], audit_results['suburban_fringe']['f1_score'], audit_results['suburban_fringe']['precision']]
    }, index=["Accuracy", "F1 Score", "Precision"])

    st.bar_chart(comp_df, use_container_width=True, color=["#3FA7A0", "#F2A541"])


# =====================================================================
# TAB 6: Multi-Modal Ensemble Meta-Learner (Tier 3 Stretch Goal)
# =====================================================================
with tab_multimodal:
    st.subheader("🤖 Multi-Modal Meta-Learner Fusion (Tier 3 Stretch Goal)")
    st.caption("Combines Tabular XGBoost scores (50%), PyTorch LSTM Rainfall Forecasts (30%), and Sentinel-1 SAR Water Masks (20%).")

    if st.button("⚡ Run Multi-Modal Meta-Learner Pipeline", type="primary"):
        with st.spinner("Executing PyTorch LSTM forecasting & Sentinel-1 SAR backscatter segmentation..."):
            lstm_model = train_chirps_lstm_forecaster()
            sar_df = extract_sentinel1_sar_water_mask(df_towers)
            meta_df = compute_multimodal_ensemble(df_towers, lstm_model, sar_df)

            st.success("Multi-Modal Meta-Learner Pipeline Completed!")

            disp_meta = meta_df[["tower_id", "district", "flood_probability", "lstm_forecast_3d_mm", "sar_water_detected", "meta_ensemble_score", "meta_risk_tier"]]
            st.dataframe(
                disp_meta.style.format({"flood_probability": "{:.2f}", "lstm_forecast_3d_mm": "{:.1f}mm", "meta_ensemble_score": "{:.2f}"}),
                use_container_width=True,
                height=400
            )


# =====================================================================
# Footer Attribution & Metadata
# =====================================================================
st.divider()
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.80rem; color: #9BA8B5; padding-top: 4px;">
    <div>
        🛡️ <b>TeleShield</b> — ASEAN GeoAI Fusion 2026 Hackathon Prototype
    </div>
    <div>
        <span>{GEE_ATTRIBUTION}</span> &bull; 
        <span>{OPENCELLID_ATTRIBUTION}</span> &bull; 
        <a href="https://github.com/aliibtisam1001/TeleShield-GeoAI-Telecom-Tower-Flood-Resiliency-System" target="_blank" style="color: #3FA7A0; text-decoration: none; font-weight: 600;">GitHub Source</a>
    </div>
</div>
""", unsafe_allow_html=True)

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from maintenance_ml import (
    MACHINES,
    load_bundle,
    load_horizon_bundle,
    load_machine_df,
    suggest_fix,
    build_history_input_frame,
    infer_boiler_fault_type_rules,
    infer_genset_fault_type_rules,
    infer_pellet_fault_type_rules,
)

st.set_page_config(
    page_title="AI Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
)

# =========================================================
# Custom CSS
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 1rem;
    max-width: 1450px;
}
.big-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    line-height: 1.25;
    margin-top: 0.2rem;
    margin-bottom: 0.35rem;
    color: #F8FAFC;
    text-shadow: 0 2px 8px rgba(0,0,0,0.35);
}
.subtle {
    color: rgba(255,255,255,0.92);
    font-size: 1rem;
    margin-bottom: 1rem;
    text-shadow: 0 1px 6px rgba(0,0,0,0.28);
}
.hero {
    border-radius: 22px;
    padding: 1.3rem 1.5rem;
    background: linear-gradient(135deg, #2f3658 0%, #232946 45%, #161b31 100%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 32px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}
.card {
    border-radius: 18px;
    padding: 18px 18px;
    background: linear-gradient(135deg, rgba(40,40,60,0.92), rgba(15,15,25,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.kpi-card {
    border-radius: 18px;
    padding: 16px 16px;
    background: linear-gradient(135deg, rgba(32,37,64,0.94), rgba(17,21,36,0.96));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    min-height: 140px;
}
.kpi-label {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.78);
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
    color: #FFFFFF;
}
.kpi-meta {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.72);
}
.badge {
    display:inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.90rem;
    border: 1px solid rgba(255,255,255,0.12);
}
.badge-ok { background: rgba(46, 204, 113, 0.18); }
.badge-mid { background: rgba(241, 196, 15, 0.18); }
.badge-bad { background: rgba(231, 76, 60, 0.18); }
.small-note {
    font-size: 0.86rem;
    color: rgba(255,255,255,0.80);
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #F8FAFC;
}
hr.soft {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 0.8rem 0 1rem 0;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stNumberInput input {
    text-align: left;
}
.stDataFrame, .stTable {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def normalize_col_name(col_name: str) -> str:
    name = str(col_name).strip().upper()
    name = re.sub(r"__\d+$", "", name)
    return name


def is_time_like_column(col_name: str) -> bool:
    name = normalize_col_name(col_name)
    if name in {"DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME", "DAILY CONDITION", "WEEKLY CONDITION"}:
        return True
    if name.startswith("DAILY CONDITION") or name.startswith("WEEKLY CONDITION"):
        return True
    prefixes = ("DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME")
    return any(name.startswith(prefix) for prefix in prefixes)


def is_numericish_column(col_name: str) -> bool:
    name = normalize_col_name(col_name)
    numeric_keywords = [
        "OPERATING HOURS", "HOURS", "RUNNING HOURS", "RUNTIME", "LOAD",
        "TEMP", "TEMPERATURE", "PRESSURE", "CURRENT", "VOLTAGE",
        "SPEED", "RPM", "FLOW", "LEVEL", "AMPS", "KW", "KVA", "HZ", "PF",
        "WATER", "FUEL"
    ]
    return any(keyword in name for keyword in numeric_keywords)


def is_dropdown_column(machine: str, col_name: str) -> bool:
    name = normalize_col_name(col_name)

    if machine == "pellet" and name == "FEED TYPE":
        return True

    if machine == "genset" and name == "REMARKS":
        return True

    return False


def badge_class(condition: str) -> str:
    if condition == "Normal":
        return "badge badge-ok"
    if condition == "Trending to Fault":
        return "badge badge-mid"
    return "badge badge-bad"


def horizon_badge_class(label: str) -> str:
    if label == "Safe":
        return "badge badge-ok"
    if label == "Risk":
        return "badge badge-bad"
    return "badge"


def render_metric_card(label: str, value_html: str, meta: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value_html}</div>
            <div class="kpi-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_display_base_fields(feature_cols: List[str]) -> List[str]:
    base_fields = []
    for c in feature_cols:
        base = re.sub(r"__(lag|diff)\d+$", "", str(c))
        base = base.replace("__rollmean3", "").replace("__rollstd3", "")
        if base not in base_fields and not is_time_like_column(base):
            base_fields.append(base)
    return base_fields


def infer_machine_group(machine: str, col_name: str) -> str:
    c = str(col_name).upper()

    if machine == "boiler":
        if "840" in c or "WATER TEMP" in c or "FEED WATER TANK" in c:
            return "Feed Water Tank"

        if "OPERATING HOURS" in c:
            return "Operation"

        if "FUEL PRESSURE" in c or "BURNER UNIT" in c:
            return "Burner Unit"

        if "MAIN STREAM PRESSURE" in c:
            return "Boiler Unit"
        if "FUEL GAS" in c:
            return "Boiler Unit"
        if "BOILER UNIT" in c:
            return "Boiler Unit"
        if "WATER" in c and "TEMP" not in c and "840" not in c and "FEED WATER TANK" not in c:
            return "Boiler Unit"

        return "Other"

    if machine == "genset":
        if "VOLTAGE" in c:
            return "Voltage"
        if "CURRENT" in c:
            return "Current"
        if "LOAD" in c:
            return "Load"
        if "POWER FACTOR" in c:
            return "Power Factor"
        if "ENGINE" in c or "OIL" in c or "COOLANT" in c or "FREQUENCY" in c:
            return "Engine"
        if "OPERATING HOURS" in c:
            return "Operation"
        if "REMARKS" in c:
            return "Remarks"
        return "Other"

    if machine == "pellet":
        if "STEAM" in c or "US PRESS" in c or "DS PRESS" in c:
            return "Steam"
        if "PELLET" in c or "MILL" in c or "FEED TYPE" in c or "FEEDER RATE" in c or "TEMPERATURE" in c:
            return "Pellet Mill"
        if "AMP1" in c or "AMP2" in c or "MOTOR" in c:
            return "Motor"
        if "OPERATING HOURS" in c or c.strip().upper() == "HOURS":
            return "Operation"
        return "Other"


def sort_group_fields(machine: str, group: str, fields: List[str]) -> List[str]:
    if machine == "boiler" and group == "Boiler Unit":
        preferred = ["WATER", "MAIN STREAM PRESSURE", "FUEL GAS"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "boiler" and group == "Burner Unit":
        preferred = ["FUEL PRESSURE"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "boiler" and group == "Feed Water Tank":
        preferred = ["WATER", "WATER TEMP"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "pellet" and group == "Pellet Mill":
        preferred = ["FEED TYPE", "FEEDER RATE", "TEMPERATURE"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "pellet" and group == "Motor":
        preferred = ["AMP1", "AMP2"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "pellet" and group == "Steam":
        preferred = ["US PRESS", "DS PRESS"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p in x.upper()), len(preferred))
        )

    if machine == "pellet" and group == "Operation":
        preferred = ["HOURS", "OPERATING HOURS"]
        return sorted(
            fields,
            key=lambda x: next((i for i, p in enumerate(preferred) if p == x.upper()), len(preferred))
        )

    return fields


def get_dropdown_options(df_ref: pd.DataFrame, field: str) -> List[str]:
    if field not in df_ref.columns:
        return [""]

    vals = (
        df_ref[field]
        .dropna()
        .astype(str)
        .str.strip()
    )
    vals = vals[vals != ""].unique().tolist()
    vals = sorted(vals)

    if not vals:
        vals = [""]

    return vals


def build_grouped_input_form(
    machine: str,
    base_fields: List[str],
    history_steps: int,
    df_ref: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = []

    if history_steps == 1:
        labels = ["Current Day"]
    else:
        labels = [f"Day -{history_steps - 1 - i}" for i in range(history_steps - 1)] + ["Current Day"]

    for row_idx in range(history_steps):
        row_inputs: Dict[str, Any] = {}

        with st.expander(labels[row_idx], expanded=(row_idx == history_steps - 1)):
            groups: Dict[str, List[str]] = {}
            for c in base_fields:
                g = infer_machine_group(machine, c)
                groups.setdefault(g, []).append(c)

            ordered_groups = [
                g for g in [
                    "Boiler Unit", "Burner Unit", "Feed Water Tank",
                    "Voltage", "Current", "Load", "Power Factor", "Engine",
                    "Steam", "Pellet Mill", "Motor",
                    "Operation", "Remarks", "Other"
                ] if g in groups
            ]

            for group in ordered_groups:
                st.markdown(f"**{group}**")
                c1, c2 = st.columns(2, gap="large")

                group_fields = sort_group_fields(machine, group, groups[group])

                for i, field in enumerate(group_fields):
                    target_col = c1 if i % 2 == 0 else c2
                    with target_col:
                        if is_dropdown_column(machine, field):
                            options = get_dropdown_options(df_ref, field)
                            val = st.selectbox(
                                field,
                                options=options,
                                index=0,
                                key=f"{machine}_{row_idx}_{field}",
                            )
                            row_inputs[field] = val

                        elif is_numericish_column(field):
                            val = st.number_input(
                                label=field,
                                value=0.0,
                                step=0.01,
                                format="%.4f",
                                key=f"{machine}_{row_idx}_{field}",
                            )
                            row_inputs[field] = val

                        else:
                            row_inputs[field] = st.text_input(
                                field,
                                value="",
                                key=f"{machine}_{row_idx}_{field}",
                            )

                st.markdown('<hr class="soft">', unsafe_allow_html=True)

        rows.append(row_inputs)

    return rows


def get_best_pipe(bundle: Dict[str, Any], section: str):
    best_name = bundle[section]["best_model"]
    if best_name is None:
        return None, None
    return best_name, bundle[section]["pipelines"][best_name]

def final_machine_condition(ff_pred: str, horizon_pred: Optional[str]) -> str:
    ff = str(ff_pred).strip().lower()
    hz = str(horizon_pred).strip().lower() if horizon_pred is not None else ""

    if ff == "fault":
        return "Critical"
    if hz == "risk":
        return "Trending to Fault"
    return "Normal"


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def build_horizon_input_frame(machine: str, input_rows: List[Dict[str, Any]], horizon_bundle: Dict[str, Any]) -> pd.DataFrame:
    seq_len = int(horizon_bundle.get("seq_len", 7))
    trained_cols = horizon_bundle["feature_columns"]

    if len(input_rows) != seq_len:
        raise ValueError(f"Horizon model expects {seq_len} rows, but received {len(input_rows)} rows.")

    rows = []
    for row in input_rows:
        if machine == "boiler":
            base_map = {
                "BOILER_WATER_mean": _safe_float(row.get("BOILER WATER", 0.0)),
                "BOILER_WATER_min": _safe_float(row.get("BOILER WATER", 0.0)),
                "BOILER_WATER_max": _safe_float(row.get("BOILER WATER", 0.0)),
                "BOILER_WATER_std": 0.0,
                "BOILER_WATER_last": _safe_float(row.get("BOILER WATER", 0.0)),

                "MAIN_STEAM_PRESSURE_mean": _safe_float(row.get("BOILER MAIN STEAM PRESSURE", 0.0)),
                "MAIN_STEAM_PRESSURE_min": _safe_float(row.get("BOILER MAIN STEAM PRESSURE", 0.0)),
                "MAIN_STEAM_PRESSURE_max": _safe_float(row.get("BOILER MAIN STEAM PRESSURE", 0.0)),
                "MAIN_STEAM_PRESSURE_std": 0.0,
                "MAIN_STEAM_PRESSURE_last": _safe_float(row.get("BOILER MAIN STEAM PRESSURE", 0.0)),

                "FUEL_GAS_TEMP_mean": _safe_float(row.get("BOILER FUEL GAS", 0.0)),
                "FUEL_GAS_TEMP_min": _safe_float(row.get("BOILER FUEL GAS", 0.0)),
                "FUEL_GAS_TEMP_max": _safe_float(row.get("BOILER FUEL GAS", 0.0)),
                "FUEL_GAS_TEMP_std": 0.0,
                "FUEL_GAS_TEMP_last": _safe_float(row.get("BOILER FUEL GAS", 0.0)),

                "BURNER_FUEL_PRESSURE_mean": _safe_float(row.get("BURNER FUEL PRESSURE", 0.0)),
                "BURNER_FUEL_PRESSURE_min": _safe_float(row.get("BURNER FUEL PRESSURE", 0.0)),
                "BURNER_FUEL_PRESSURE_max": _safe_float(row.get("BURNER FUEL PRESSURE", 0.0)),
                "BURNER_FUEL_PRESSURE_std": 0.0,
                "BURNER_FUEL_PRESSURE_last": _safe_float(row.get("BURNER FUEL PRESSURE", 0.0)),

                "FEED_WATER_TANK_LEVEL_mean": _safe_float(row.get("FEED WATER TANK", 0.0)),
                "FEED_WATER_TANK_LEVEL_min": _safe_float(row.get("FEED WATER TANK", 0.0)),
                "FEED_WATER_TANK_LEVEL_max": _safe_float(row.get("FEED WATER TANK", 0.0)),
                "FEED_WATER_TANK_LEVEL_std": 0.0,
                "FEED_WATER_TANK_LEVEL_last": _safe_float(row.get("FEED WATER TANK", 0.0)),

                "FEED_WATER_TEMP_mean": _safe_float(row.get("FEED WATER TEMP", 0.0)),
                "FEED_WATER_TEMP_min": _safe_float(row.get("FEED WATER TEMP", 0.0)),
                "FEED_WATER_TEMP_max": _safe_float(row.get("FEED WATER TEMP", 0.0)),
                "FEED_WATER_TEMP_std": 0.0,
                "FEED_WATER_TEMP_last": _safe_float(row.get("FEED WATER TEMP", 0.0)),
            }
        elif machine == "pellet":
            base_map = {
                "AMP1_mean": _safe_float(row.get("AMP1 (100AMP)", 0.0)),
                "AMP1_min": _safe_float(row.get("AMP1 (100AMP)", 0.0)),
                "AMP1_max": _safe_float(row.get("AMP1 (100AMP)", 0.0)),
                "AMP1_std": 0.0,
                "AMP1_last": _safe_float(row.get("AMP1 (100AMP)", 0.0)),

                "AMP2_mean": _safe_float(row.get("AMP2 (100AMP)", 0.0)),
                "AMP2_min": _safe_float(row.get("AMP2 (100AMP)", 0.0)),
                "AMP2_max": _safe_float(row.get("AMP2 (100AMP)", 0.0)),
                "AMP2_std": 0.0,
                "AMP2_last": _safe_float(row.get("AMP2 (100AMP)", 0.0)),

                "US_PRESS_mean": _safe_float(row.get("US PRESS (4.5-8BARS)", 0.0)),
                "US_PRESS_min": _safe_float(row.get("US PRESS (4.5-8BARS)", 0.0)),
                "US_PRESS_max": _safe_float(row.get("US PRESS (4.5-8BARS)", 0.0)),
                "US_PRESS_std": 0.0,
                "US_PRESS_last": _safe_float(row.get("US PRESS (4.5-8BARS)", 0.0)),

                "DS_PRESS_mean": _safe_float(row.get("DS PRESS (1.8-3BARS)", 0.0)),
                "DS_PRESS_min": _safe_float(row.get("DS PRESS (1.8-3BARS)", 0.0)),
                "DS_PRESS_max": _safe_float(row.get("DS PRESS (1.8-3BARS)", 0.0)),
                "DS_PRESS_std": 0.0,
                "DS_PRESS_last": _safe_float(row.get("DS PRESS (1.8-3BARS)", 0.0)),

                "FEEDER_RATE_mean": _safe_float(row.get("FEEDER RATE (60HERTZ) - (%)", 0.0)),
                "FEEDER_RATE_min": _safe_float(row.get("FEEDER RATE (60HERTZ) - (%)", 0.0)),
                "FEEDER_RATE_max": _safe_float(row.get("FEEDER RATE (60HERTZ) - (%)", 0.0)),
                "FEEDER_RATE_std": 0.0,
                "FEEDER_RATE_last": _safe_float(row.get("FEEDER RATE (60HERTZ) - (%)", 0.0)),

                "TEMPERATURE_mean": _safe_float(row.get("TEMPERATURE", 0.0)),
                "TEMPERATURE_min": _safe_float(row.get("TEMPERATURE", 0.0)),
                "TEMPERATURE_max": _safe_float(row.get("TEMPERATURE", 0.0)),
                "TEMPERATURE_std": 0.0,
                "TEMPERATURE_last": _safe_float(row.get("TEMPERATURE", 0.0)),
            }
        else:
            raise ValueError(f"Horizon model not supported for machine: {machine}")

        rows.append(base_map)

    X_row = {}
    keys = list(rows[0].keys())

    for step_idx in range(seq_len):
        suffix = f"d{step_idx + 1}"
        step = rows[step_idx]
        for col, val in step.items():
            X_row[f"{col}__{suffix}"] = val

    for base_col in keys:
        series_vals = [r[base_col] for r in rows]
        X_row[f"{base_col}__delta"] = float(series_vals[-1] - series_vals[0])
        X_row[f"{base_col}__wmean"] = float(np.mean(series_vals))
        X_row[f"{base_col}__wstd"] = float(np.std(series_vals))

    Xh = pd.DataFrame([X_row])

    for c in trained_cols:
        if c not in Xh.columns:
            Xh[c] = np.nan

    Xh = Xh[trained_cols].copy()
    return Xh


def predict_fault_horizon(machine: str, input_rows: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[float], str]:
    if machine not in {"boiler", "pellet"}:
        return None, None, "Not applicable"

    horizon_bundle = load_horizon_bundle(machine)
    if not horizon_bundle:
        return None, None, "Horizon model not available"

    try:
        best_name = horizon_bundle["best_model"]
        pipe = horizon_bundle["pipelines"][best_name]
        Xh = build_horizon_input_frame(machine, input_rows, horizon_bundle)

        pred = str(pipe.predict(Xh)[0])

        conf = None
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(Xh)[0]
            conf = float(np.max(prob))

        return pred, conf, best_name
    except Exception as e:
        return None, None, f"Horizon prediction unavailable: {e}"


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="big-title">🛠️ AI Predictive Maintenance Dashboard</div>
        <div class="subtle">
            Predict machine condition, fault type, and fault horizon risk.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Prediction Controls")

machine = st.sidebar.selectbox(
    "Select Machine",
    ["boiler", "genset", "pellet"],
    format_func=lambda x: x.upper(),
)

bundle = load_bundle(machine)
if not bundle:
    st.sidebar.error("No trained models found. Run: python train_models.py")
    st.stop()

cfg = MACHINES[machine]
history_steps = 3 if machine in {"boiler", "pellet"} else 1

st.sidebar.markdown("---")
if machine in {"boiler", "pellet"}:
    st.sidebar.caption("Boiler and Pellet use 3 input sets: Day -2, Day -1, and Current Day. Fault horizon also uses the same 3-day window.")
else:
    st.sidebar.caption("Genset uses current values only.")

# =========================================================
# Reference
# =========================================================
df_ref = load_machine_df(machine)
feature_cols = bundle["feature_columns"]
base_fields = get_display_base_fields(feature_cols)

if machine == "boiler":
    base_fields = [f for f in base_fields if normalize_col_name(f) != "OPERATING HOURS"]
elif machine == "pellet":
    base_fields = [f for f in base_fields if normalize_col_name(f) not in {"HOURS", "OPERATING HOURS"}]

# =========================================================
# Top summary
# =========================================================
ctop1, ctop2, ctop3 = st.columns([1.15, 1.15, 1], gap="large")
with ctop1:
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Machine Selected</div>
            <div class="small-note">{machine.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ctop2:
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Base Input Fields</div>
            <div class="small-note">{len(base_fields)} user-entered fields</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ctop3:
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Fault Horizon Classes</div>
            <div class="small-note">Safe / Risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# =========================================================
# Input form
# =========================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔎 Input Machine Features")

if machine in ["boiler", "pellet"]:
    st.caption("Enter exactly 3 sets of values in order: Day -2, Day -1, and Current Day.")
else:
    st.caption("Enter the current/latest values only.")

with st.form("prediction_form"):
    input_rows = build_grouped_input_form(machine, base_fields, history_steps, df_ref)
    predict_btn = st.form_submit_button("🚀 Predict Machine Status", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Prediction
# =========================================================
if predict_btn:
    try:
        X = build_history_input_frame(machine, input_rows, bundle)

        ff_name, ff_pipe = get_best_pipe(bundle, "fault_flag")
        ff_pred = str(ff_pipe.predict(X)[0]).strip()

        horizon_pred = None
        horizon_conf = None
        horizon_model_name = "N/A"

        if machine in {"boiler", "pellet"}:
            horizon_pred, horizon_conf, horizon_model_name = predict_fault_horizon(machine, input_rows)

        final_condition = final_machine_condition(ff_pred, horizon_pred)

        fault_type_pred = "N/A"
        fault_type_model_name = "Rule-based" if cfg["fault_type_mode"] == "rule_based" else "ML / Rule Fallback"

        if final_condition != "Normal":
            if machine == "boiler":
                ft = infer_boiler_fault_type_rules(input_rows)
                fault_type_pred = ft if ft else "N/A"

            elif machine == "genset":
                ft = infer_genset_fault_type_rules(input_rows[-1])
                fault_type_pred = ft if ft else "N/A"

            elif machine == "pellet":
                rule_ft = infer_pellet_fault_type_rules(input_rows)
                if bundle["fault_type"]["pipelines"] and bundle["fault_type"]["best_model"]:
                    best_ft = bundle["fault_type"]["best_model"]
                    ft_pipe = bundle["fault_type"]["pipelines"][best_ft]
                    fault_type_model_name = f"{best_ft} + rule fallback"
                    fault_type_pred = str(ft_pipe.predict(X)[0])
                    if not fault_type_pred or fault_type_pred in {"Normal", "N/A"}:
                        fault_type_pred = rule_ft if rule_ft else "N/A"
                else:
                    fault_type_model_name = "Rule-based"
                    fault_type_pred = rule_ft if rule_ft else "N/A"

        fixes = suggest_fix(machine, fault_type_pred)

        st.write("")
        k1, k2, k3 = st.columns(3, gap="large")

        with k1:
            render_metric_card(
                "Machine Condition",
                f'<span class="{badge_class(final_condition)}">{final_condition}</span>',
                f"Fault flag model: {ff_name} | Horizon model: {horizon_model_name if machine in {'boiler', 'pellet'} else 'N/A'}",
            )

        with k2:
            render_metric_card(
                "Fault Type",
                fault_type_pred,
                f"Method: {fault_type_model_name}",
            )

        with k3:
            if str(ff_pred).strip().lower() == "fault":
                value_text = "Disregarded"
                meta_text = "Fault detected by fault flag model; final condition automatically set to Critical"
            elif horizon_pred is None:
                value_text = "N/A"
                meta_text = horizon_model_name
            else:
                value_text = f'<span class="{horizon_badge_class(horizon_pred)}">{horizon_pred}</span>'
                meta_text = horizon_model_name if horizon_conf is None else f"{horizon_model_name} | Confidence: {horizon_conf:.2%}"

            render_metric_card(
                "Fault Horizon",
                value_text,
                meta_text,
            )

        st.write("")
        lower1, lower2 = st.columns([1.25, 1], gap="large")

        with lower1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧰 Suggested Fix / Recommended Actions")
            for i, line in enumerate(fixes, 1):
                st.write(f"{i}. {line}")
            st.markdown("</div>", unsafe_allow_html=True)

        with lower2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📌 Prediction Summary")
            st.write(f"**Machine:** {machine.upper()}")
            st.write(f"**Machine Condition:** {final_condition}")
            st.write(f"**Fault Type:** {fault_type_pred}")

            if str(ff_pred).strip().lower() == "fault":
                st.write("**Fault Horizon:** Disregarded")
                st.write("**Horizon Confidence:** N/A")
            else:
                st.write(f"**Fault Horizon:** {horizon_pred if horizon_pred is not None else 'N/A'}")
                st.write(f"**Horizon Confidence:** {'N/A' if horizon_conf is None else f'{horizon_conf:.2%}'}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Tabular Model Input Used")
        st.dataframe(X, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
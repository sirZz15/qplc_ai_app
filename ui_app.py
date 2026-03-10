from __future__ import annotations

import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit as st
import sys
import platform
import sklearn
import pandas as pd
import numpy as np
import scipy
import joblib

st.sidebar.title("Deployment Environment")

st.sidebar.write("### Python")
st.sidebar.write(sys.version)

st.sidebar.write("### Platform")
st.sidebar.write(platform.platform())

st.sidebar.write("### Libraries")
st.sidebar.write("streamlit:", st.__version__)
st.sidebar.write("scikit-learn:", sklearn.__version__)
st.sidebar.write("pandas:", pd.__version__)
st.sidebar.write("numpy:", np.__version__)
st.sidebar.write("scipy:", scipy.__version__)
st.sidebar.write("joblib:", joblib.__version__)
from maintenance_ml import (
    load_bundle,
    load_machine_df,
    suggest_fix,
)

st.set_page_config(
    page_title="QPLC AI Predictive Maintenance Dashboard",
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
TIME_LIKE_COLS = {
    "DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME",
    "DAILY CONDITION", "DAILY CONDITION_1"
}


def normalize_col_name(col_name: str) -> str:
    name = str(col_name).strip().upper()
    name = re.sub(r"__\d+$", "", name)
    return name


def is_time_like_column(col_name: str) -> bool:
    name = normalize_col_name(col_name)

    # Explicit exclusions
    if name in TIME_LIKE_COLS:
        return True

    # Exclude daily tracking / daily condition fields
    if name.startswith("DAILY CONDITION"):
        return True

    prefixes = ("DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME")
    return any(name.startswith(prefix) for prefix in prefixes)


def is_forced_numeric_column(col_name: str) -> bool:
    name = normalize_col_name(col_name)

    numeric_keywords = [
        "OPERATING HOURS",
        "HOURS",
        "RUNNING HOURS",
        "RUNTIME",
        "LOAD",
        "TEMP",
        "TEMPERATURE",
        "PRESSURE",
        "CURRENT",
        "VOLTAGE",
        "SPEED",
        "RPM",
        "FLOW",
        "LEVEL",
        "AMPS",
        "KW",
        "KVA",
        "HZ",
        "PF",
    ]

    return any(keyword in name for keyword in numeric_keywords) or name == "HOURS"


def clean_display_columns(cols: List[str]) -> List[str]:
    return [c for c in cols if not is_time_like_column(c)]


def badge_class(sev: str) -> str:
    if sev == "Non-Critical":
        return "badge badge-ok"
    if sev == "Inconvenient":
        return "badge badge-mid"
    return "badge badge-bad"


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


def infer_machine_group(machine: str, col_name: str) -> str:
    c = str(col_name).upper()

    if machine == "boiler":
        if "BOILER UNIT" in c:
            return "Boiler Unit"
        if "BURNER UNIT" in c:
            return "Burner Unit"
        if "FEED WATER TANK" in c:
            return "Feed Water Tank"
        if "OPERATING HOURS" in c:
            return "Operation"
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
        if "ENGINE" in c:
            return "Engine"
        if "OPERATING HOURS" in c:
            return "Operation"
        if "REMARKS" in c:
            return "Remarks"
        return "Other"

    if machine == "pellet":
        if "STEAM" in c:
            return "Steam"
        if "PELLET" in c or "MILL" in c:
            return "Pellet Mill"
        if "MOTOR" in c:
            return "Motor"
        if "OPERATING HOURS" in c or c.strip().upper() == "HOURS":
            return "Operation"
        return "Other"

    return "Other"


def build_grouped_input_form(df_ref: pd.DataFrame, machine: str, cols: List[str]) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    cols = clean_display_columns(cols)

    groups: Dict[str, List[str]] = {}
    for c in cols:
        g = infer_machine_group(machine, c)
        groups.setdefault(g, []).append(c)

    preferred_order = [
        "Boiler Unit", "Burner Unit", "Feed Water Tank",
        "Voltage", "Current", "Load", "Power Factor", "Engine",
        "Steam", "Pellet Mill", "Motor",
        "Operation", "Remarks", "Other"
    ]
    ordered_groups = [g for g in preferred_order if g in groups] + [g for g in groups if g not in preferred_order]

    for group in ordered_groups:
        st.markdown(f"**{group}**")
        col1, col2 = st.columns(2, gap="large")
        group_cols = groups[group]

        for i, c in enumerate(group_cols):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                s = df_ref[c] if c in df_ref.columns else pd.Series(dtype="object")

                s_num = pd.to_numeric(s, errors="coerce")
                numeric_ratio = s_num.notna().mean() if len(s) else 0.0

                # Force numeric input for operating hours and similar numeric fields
                if is_forced_numeric_column(c) or pd.api.types.is_numeric_dtype(s) or numeric_ratio >= 0.50:
                    default_val = float(np.nanmedian(s_num.values)) if s_num.notna().any() else 0.0
                    val = st.number_input(
                        label=c,
                        value=default_val,
                        step=0.1,
                        format="%.4f",
                        key=f"grouped_{c}",
                    )
                    inputs[c] = val
                else:
                    uniq = s.dropna().astype(str).str.strip()
                    uniq = uniq[uniq != ""]
                    options = ["(blank)"] + uniq.value_counts().head(15).index.tolist()
                    sel = st.selectbox(c, options, key=f"grouped_{c}")
                    inputs[c] = "" if sel == "(blank)" else sel

        st.markdown('<hr class="soft">', unsafe_allow_html=True)

    return inputs


# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="big-title">🛠️ QPLC AI Predictive Maintenance Dashboard</div>
        <div class="subtle">
            Input machine readings to predict machine condition, fault status,
            fault type, remaining useful life, and recommended corrective action.
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

raw_feature_cols = bundle["feature_columns"]
feature_cols = [c for c in raw_feature_cols if not is_time_like_column(c)]

use_rul = st.sidebar.checkbox("Predict Lifespan (RUL) if available", value=True)
group_inputs = st.sidebar.checkbox("Group inputs by subsystem", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("This app is configured for prediction only.")

# =========================================================
# Top summary cards
# =========================================================
df_ref = load_machine_df(machine)

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
            <div class="section-title">Input Fields Used</div>
            <div class="small-note">{len(feature_cols)} predictive features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with ctop3:
    best_rul_text = bundle["rul"]["best_model"] if bundle.get("rul") else "Not Available"
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Best RUL Model</div>
            <div class="small-note">{best_rul_text}</div>
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
st.caption("Time-index fields such as DAY, DAY__1, WEEK, DAILY CONDITION, and similar columns are excluded from the form.")

with st.expander("Show / Hide Input Form", expanded=True):
    inputs = build_grouped_input_form(df_ref, machine, feature_cols)

st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_btn = st.button("🚀 Predict Machine Status", use_container_width=True)

# =========================================================
# Prediction
# =========================================================
if predict_btn:
    X = pd.DataFrame([{
        c: inputs.get(c, np.nan) if not is_time_like_column(c) else np.nan
        for c in raw_feature_cols
    }])

    sev_name = bundle["severity"]["best_model"]
    ff_name = bundle["fault_flag"]["best_model"]

    sev_pipe = bundle["severity"]["pipelines"][sev_name]
    ff_pipe = bundle["fault_flag"]["pipelines"][ff_name]

    sev_pred = str(sev_pipe.predict(X)[0])
    ff_pred = str(ff_pipe.predict(X)[0])

    fault_type_pred = "Normal"
    fault_type_model_name = "N/A"

    if ff_pred == "Fault":
        if bundle["fault_type"]["pipelines"]:
            best_ft = bundle["fault_type"]["best_model"]
            if best_ft:
                ft_pipe = bundle["fault_type"]["pipelines"][best_ft]
                fault_type_model_name = best_ft
                fault_type_pred = str(ft_pipe.predict(X)[0])
            else:
                fault_type_pred = "Fault (unspecified)"
        else:
            fault_type_pred = "Fault (unspecified)"

    rul_hours = None
    rul_model_name = "N/A"

    if use_rul and bundle.get("rul"):
        try:
            best_rul = bundle["rul"]["best_model"]
            rul_model_name = best_rul
            rul_pipe = bundle["rul"]["pipelines"][best_rul]
            val = float(rul_pipe.predict(X)[0])
            if np.isfinite(val) and val >= 0:
                rul_hours = val
        except Exception:
            rul_hours = None

    fixes = suggest_fix(machine, fault_type_pred)

    st.write("")
    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        render_metric_card(
            "Machine Condition",
            f'<span class="{badge_class(sev_pred)}">{sev_pred}</span>',
            f"Model used: {sev_name}",
        )

    with k2:
        ff_badge = "badge badge-ok" if ff_pred == "Normal" else "badge badge-bad"
        render_metric_card(
            "Machine Fault",
            f'<span class="{ff_badge}">{ff_pred}</span>',
            f"Model used: {ff_name}",
        )

    with k3:
        render_metric_card(
            "Possible Fault",
            fault_type_pred,
            f"Model used: {fault_type_model_name}",
        )

    with k4:
        rul_text = "N/A" if rul_hours is None else f"{rul_hours:,.2f} hours"
        rul_meta = "Not trained or insufficient fault history" if rul_hours is None else f"Model used: {rul_model_name}"
        render_metric_card(
            "Estimated Lifespan (RUL)",
            rul_text,
            rul_meta,
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
        st.write(f"**Condition:** {sev_pred}")
        st.write(f"**Fault Flag:** {ff_pred}")
        st.write(f"**Fault Type:** {fault_type_pred}")
        st.write(f"**RUL:** {'N/A' if rul_hours is None else f'{rul_hours:,.2f} hours'}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Submitted Inputs")
    st.dataframe(X, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
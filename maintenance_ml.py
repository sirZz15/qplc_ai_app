from __future__ import annotations

import os
import re
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer


# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCEL_PATH = os.path.join(BASE_DIR, "data", "QPLC-Maintenance Data.xlsx")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models_qplc"))
os.makedirs(MODEL_DIR, exist_ok=True)

MACHINES = {
    "boiler": {"sheet": "Boiler", "time_key": "DAY", "condition_col": "DAILY CONDITION"},
    "genset": {"sheet": "Gen Set", "time_key": "WEEK", "condition_col": "WEEKLY CONDITION"},
    "pellet": {"sheet": "Pellet Mill", "time_key": "DAY", "condition_col": "DAILY CONDITION"},
}

# =========================================================
# Google Drive model file IDs
# Replace these with your actual Google Drive file IDs
# =========================================================
MODEL_FILE_IDS = {
    "boiler": "1wegYDB9ZgDwx0_z7ckWRvxMiuiWUlbu-",
    "genset": "13gI10UvdCAAlMj3ePa7BwAQujpZ7k14H",
    "pellet": "1wZzZhsJsPMe92brSKa98Ah5z3edlv3ES",
}

MODEL_FILES = {
    "boiler": "boiler_bundle.joblib",
    "genset": "genset_bundle.joblib",
    "pellet": "pellet_bundle.joblib",
}


# =========================
# Excel parsing (merged headers)
# =========================
def _make_unique(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def parse_merged_header_sheet(xlsx_path: str, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

    header_idx = None
    for i in range(len(raw)):
        v = raw.iloc[i, 1]
        if isinstance(v, str) and v.strip().upper() in ["DAY", "WEEK", "HOURS"]:
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    h1 = raw.iloc[header_idx].copy()
    h2 = raw.iloc[header_idx + 1].copy()

    group = h1.ffill().bfill()
    sub = h2.fillna("")

    cols = []
    for g, s in zip(group, sub):
        g = "" if pd.isna(g) else str(g).strip()
        s = str(s).strip()
        if s and g and s != g:
            col = f"{g} - {s}"
        elif g:
            col = g
        elif s:
            col = s
        else:
            col = "col"
        cols.append(col)

    data = raw.iloc[header_idx + 2:].copy()
    data.columns = _make_unique(cols)
    data = data.dropna(how="all").reset_index(drop=True)
    data = data.loc[:, [c for c in data.columns if c and c != "col"]]
    return data


# =========================
# Label engineering
# =========================
def normalize_condition_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.upper() == "NORMAL":
        return "Normal"
    return s


def map_condition_to_severity(cond: str) -> str:
    c = (cond or "").strip()
    if not c or c == "-":
        return "Non-Critical"
    if re.match(r"(?i)^\s*fault\s*:", c):
        return "Critical"
    if re.match(r"(?i)^\s*trend\s*:", c):
        return "Inconvenient"
    if re.search(r"(?i)repair|reset|post-?repair", c):
        return "Inconvenient"
    if re.match(r"(?i)^\s*normal\b", c):
        return "Non-Critical"
    return "Inconvenient"


def extract_fault_type(cond: str) -> str:
    c = (cond or "").strip()
    m = re.search(r"(?i)fault\s*:\s*(.*)", c)
    if m:
        t = m.group(1).strip()
        return t if t else "Fault"
    return "Normal"


def fault_flag_from_type(ftype: str) -> str:
    return "Fault" if (ftype and ftype != "Normal") else "Normal"


# =========================
# Suggested Fix Library (editable)
# =========================
FIX_LIBRARY = {
    "boiler": {
        "Tube Scaling": [
            "Inspect tubes for scaling; confirm via ΔT/efficiency trends.",
            "Perform chemical descaling or mechanical cleaning per OEM.",
            "Review water treatment (TDS/hardness), blowdown schedule.",
        ],
        "Main Steam Line Leak": [
            "Isolate section if safe; inspect joints/valves/gaskets.",
            "Repair/replace damaged segment and pressure test.",
        ],
        "Low Water Trip": [
            "Check feedwater supply/pump and tank level.",
            "Inspect level probes/controller; clean/recalibrate.",
        ],
    },
    "genset": {
        "Low Oil Pressure": [
            "Check oil level/leaks; top up with correct grade.",
            "Inspect oil filter/pump; verify pressure sensor calibration.",
            "Check bearing wear if persistent.",
        ],
        "Overheating": [
            "Check coolant level, radiator condition; clean fins and ensure airflow.",
            "Inspect thermostat/water pump; verify fan operation.",
        ],
    },
    "pellet": {
        "PRV Malfunction": [
            "Inspect PRV for sticking/contamination; clean/replace.",
            "Verify pressure sensors and setpoints; check steam traps/strainers.",
        ],
        "Motor Overload": [
            "Inspect die/roller condition and lubrication; check for binding.",
            "Reduce feed rate; verify motor current vs rating; check overload relay.",
        ],
    },
}


def suggest_fix(machine: str, fault_type: str) -> List[str]:
    if not fault_type or fault_type == "Normal":
        return ["No action required. Continue monitoring and preventive maintenance."]
    lib = FIX_LIBRARY.get(machine, {})
    if fault_type in lib:
        return lib[fault_type]
    ft = fault_type.lower()
    if "oil" in ft:
        return ["Check oil level/quality, filters, pump, and sensor calibration."]
    if "overheat" in ft or "temperature" in ft:
        return ["Inspect cooling/ventilation and load; validate temperature sensors."]
    if "pressure" in ft:
        return ["Verify valves/filters, pressure sensors, and supply stability."]
    return ["Perform targeted checks per OEM manual; validate sensors and setpoints."]


# =========================
# RUL (Remaining Useful Life) target
# =========================
def build_rul_target(df: pd.DataFrame, machine: str, time_key: str, cond_col: str) -> pd.DataFrame:
    out = df.copy()

    if time_key in out.columns:
        out[time_key] = out[time_key].ffill()
        out[time_key] = pd.to_numeric(out[time_key], errors="coerce")
    else:
        out[time_key] = np.arange(len(out), dtype=float)

    out[cond_col] = out[cond_col].apply(normalize_condition_text)

    if machine == "pellet" and "HOURS" in out.columns:
        hrs = pd.to_numeric(out["HOURS"], errors="coerce").fillna(0.0)
    elif "OPERATING HOURS" in out.columns:
        hrs = pd.to_numeric(out["OPERATING HOURS"], errors="coerce").fillna(0.0)
    else:
        hrs = pd.Series(np.ones(len(out)), index=out.index)

    out["_hrs"] = hrs.clip(lower=0.0)
    out = out.sort_values(by=time_key).reset_index(drop=True)
    out["_cum_hours"] = out["_hrs"].cumsum()

    fault_type = out[cond_col].apply(extract_fault_type)
    out["_is_fault"] = (fault_type != "Normal").astype(int)

    next_fault_cum = np.full(len(out), np.nan)
    next_idx = None
    for i in range(len(out) - 1, -1, -1):
        if out.loc[i, "_is_fault"] == 1:
            next_idx = i
        if next_idx is not None:
            next_fault_cum[i] = out.loc[next_idx, "_cum_hours"]

    out["rul_hours"] = next_fault_cum - out["_cum_hours"]
    out.loc[out["rul_hours"] < 0, "rul_hours"] = np.nan
    return out


# =========================
# Preprocessing
# =========================
def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = []
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return pre, numeric_cols, cat_cols


# =========================
# Load machine dataset
# =========================
def load_machine_df(machine: str) -> pd.DataFrame:
    cfg = MACHINES[machine]
    df = parse_merged_header_sheet(EXCEL_PATH, cfg["sheet"])
    cond_col = cfg["condition_col"]

    df[cond_col] = df[cond_col].apply(normalize_condition_text)
    df["condition_severity"] = df[cond_col].apply(map_condition_to_severity)
    df["fault_type"] = df[cond_col].apply(extract_fault_type)
    df["fault_flag"] = df["fault_type"].apply(fault_flag_from_type)
    return df


# =========================
# Model artifact helpers
# =========================
def artifact_path(machine: str) -> str:
    return os.path.join(MODEL_DIR, MODEL_FILES[machine])


def save_bundle(machine: str, bundle: Dict[str, Any]) -> None:
    joblib.dump(bundle, artifact_path(machine))


def _download_from_gdrive(file_id: str, output_path: str) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def ensure_bundle_exists(machine: str) -> str:
    if machine not in MODEL_FILES:
        raise ValueError(f"Unknown machine: {machine}")

    if machine not in MODEL_FILE_IDS:
        raise ValueError(f"No Google Drive file ID configured for machine: {machine}")

    output_path = artifact_path(machine)

    if not os.path.exists(output_path):
        file_id = MODEL_FILE_IDS[machine]
        if not file_id or file_id.startswith("PASTE_YOUR_"):
            raise ValueError(
                f"Google Drive file ID for '{machine}' is not set. "
                f"Please update MODEL_FILE_IDS in maintenance.py."
            )
        _download_from_gdrive(file_id, output_path)

    return output_path


@st.cache_resource
def load_bundle(machine: str) -> Dict[str, Any]:
    p = ensure_bundle_exists(machine)
    if not os.path.exists(p):
        return {}
    return joblib.load(p)
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
    "boiler": {
        "sheet": "Boiler",
        "time_key": "DAY",
        "condition_col": "DAILY CONDITION",
        "fault_type_mode": "rule_based",
        "use_history": True,
        "rul_enabled": True,
    },
    "genset": {
        "sheet": "Gen Set",
        "time_key": "WEEK",
        "condition_col": "WEEKLY CONDITION",
        "fault_type_mode": "rule_based",
        "use_history": False,
        "rul_enabled": False,
    },
    "pellet": {
        "sheet": "Pellet Mill",
        "time_key": "DAY",
        "condition_col": "DAILY CONDITION",
        "fault_type_mode": "ml",
        "use_history": True,
        "rul_enabled": True,
    },
}

# =========================================================
# Google Drive model file IDs
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

# =========================================================
# Google Drive horizon model IDs (tabular classifier)
# =========================================================
HORIZON_MODEL_FILE_IDS = {
    "boiler": "1tJVR1e7AEAqUUxG3wyMI4tsNzkDuPp_V",
    "pellet": "1XITgMJEiRNIzQContIcWJ-QHqL9TOnTX",
}

HORIZON_MODEL_FILES = {
    "boiler": "boiler_horizon_bundle.joblib",
    "pellet": "pellet_horizon_bundle.joblib",
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
        values = raw.iloc[i].astype(str).str.upper().tolist()
        if any(v.strip() in ["DAY", "WEEK", "HOURS"] for v in values):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    h1 = raw.iloc[header_idx].fillna("").astype(str).str.strip()
    h2 = raw.iloc[header_idx + 1].fillna("").astype(str).str.strip() if header_idx + 1 < len(raw) else pd.Series([""] * len(h1))

    # If second row looks like real data, do not use it as header
    h2_numeric_ratio = pd.to_numeric(h2, errors="coerce").notna().mean()
    use_two_header_rows = h2_numeric_ratio < 0.40

    cols = []
    if use_two_header_rows:
        group = h1.replace("", np.nan).ffill().bfill()
        sub = h2

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

        data_start = header_idx + 2
    else:
        cols = [c if c else "col" for c in h1.tolist()]
        data_start = header_idx + 1

    data = raw.iloc[data_start:].copy()
    data.columns = _make_unique(cols)
    data = data.dropna(how="all").reset_index(drop=True)
    data = data.loc[:, [c for c in data.columns if c and c != "col"]]
    return data


# =========================
# Basic helpers
# =========================
def normalize_col_name(col_name: str) -> str:
    name = str(col_name).strip().upper()
    name = re.sub(r"__\d+$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def find_matching_column(df: pd.DataFrame, target: str) -> Optional[str]:
    t = normalize_col_name(target)
    for c in df.columns:
        if normalize_col_name(c) == t:
            return c
    for c in df.columns:
        if t in normalize_col_name(c):
            return c
    return None


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def clean_pellet_rows(df: pd.DataFrame, cond_col: str) -> pd.DataFrame:
    out = df.copy()

    hours_col = find_matching_column(out, "HOURS")
    amp1_col = find_matching_column(out, "AMP1")
    amp2_col = find_matching_column(out, "AMP2")
    us_col = find_matching_column(out, "US PRESS")
    ds_col = find_matching_column(out, "DS PRESS")
    feeder_col = find_matching_column(out, "FEEDER RATE")
    temp_col = find_matching_column(out, "TEMPERATURE")

    sensor_cols = [c for c in [amp1_col, amp2_col, us_col, ds_col, feeder_col, temp_col] if c is not None]

    if hours_col is None:
        return out.reset_index(drop=True)

    out[hours_col] = clean_numeric_series(out[hours_col]).fillna(0.0)

    for c in sensor_cols:
        out[c] = clean_numeric_series(out[c]).fillna(0.0)

    out[cond_col] = out[cond_col].fillna("").astype(str).str.strip()

    zero_mask = out[hours_col].eq(0)

    if sensor_cols:
        zero_sensor_mask = out[sensor_cols].sum(axis=1).eq(0)
    else:
        zero_sensor_mask = pd.Series(True, index=out.index)

    blank_condition_mask = out[cond_col].eq("") | out[cond_col].eq("Normal")
    drop_mask = zero_mask & zero_sensor_mask & blank_condition_mask

    out = out.loc[~drop_mask].copy().reset_index(drop=True)
    return out


# =========================
# Label engineering
# =========================
def normalize_condition_text(x: Any) -> str:
    if pd.isna(x):
        return "Normal"
    s = str(x).strip()
    if s == "" or s == "-":
        return "Normal"
    if s.upper() == "NORMAL":
        return "Normal"
    return s


def map_condition_to_severity(cond: str) -> str:
    c = (cond or "").strip()
    if not c or c == "-" or c.lower() == "normal":
        return "Normal"
    if re.match(r"(?i)^\s*fault\s*:", c):
        return "Fault"
    if re.match(r"(?i)^\s*trend\s*:", c):
        return "Trend"
    return "Normal"


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
# Suggested Fix Library
# =========================
FIX_LIBRARY = {
    "boiler": {
        "Tube Scaling": [
            "Inspect boiler tubes for scaling or deposits.",
            "Perform descaling or tube cleaning as required.",
            "Review feedwater quality and blowdown schedule.",
        ],
        "Main Steam Line Leak": [
            "Inspect steam line, flanges, and valves for leakage.",
            "Repair or replace defective line components.",
        ],
        "Low Water Trip": [
            "Inspect feedwater supply and tank level.",
            "Check low-water safety device and level control.",
        ],
        "High Flue Gas Temp Trip": [
            "Inspect heat transfer surfaces for fouling or scaling.",
            "Check combustion efficiency and burner settings.",
        ],
        "Burner Trip - Low Fuel Pressure": [
            "Inspect fuel supply line and fuel pump pressure.",
            "Check filter restriction or regulator malfunction.",
        ],
        "Fuel Pump Fail": [
            "Inspect fuel pump operation and power supply.",
            "Check suction blockage, leakage, and pump wear.",
        ],
        "Burner Nozzle Fault": [
            "Inspect burner nozzle for clogging or improper spray.",
            "Clean or replace nozzle and recheck combustion.",
        ],
        "Makeup Valve Stuck": [
            "Inspect makeup valve movement and control signal.",
            "Clean or replace the valve if sticking persists.",
        ],
        "Level Controller Malfunction": [
            "Check level controller calibration and sensor readings.",
            "Inspect control loop stability and output response.",
        ],
    },
    "genset": {
        "Low Oil Pressure": [
            "Check oil level and leaks.",
            "Inspect oil pump, oil filter, and sensor calibration.",
        ],
        "High Oil Pressure": [
            "Inspect oil line restriction and sensor condition.",
            "Verify oil viscosity and regulator operation.",
        ],
        "Overheating": [
            "Inspect coolant level, radiator cleanliness, and airflow.",
            "Check fan, thermostat, and pump operation.",
        ],
        "Critical Overheat": [
            "Shut down if required and inspect cooling system immediately.",
            "Check coolant circulation, radiator blockage, and load condition.",
        ],
        "Frequency Instability": [
            "Inspect governor response and load fluctuation.",
            "Check fuel supply stability and generator control settings.",
        ],
        "Voltage Drop": [
            "Inspect AVR, excitation system, and terminal connections.",
            "Check phase balance and electrical loading.",
        ],
        "Current Unbalance": [
            "Inspect three-phase load distribution.",
            "Check cable/terminal issues and abnormal branch load.",
        ],
    },
    "pellet": {
        "PRV Malfunction": [
            "Inspect pressure reducing valve for sticking or incorrect setpoint.",
            "Check steam line cleanliness and pressure control response.",
        ],
        "Motor Overload": [
            "Inspect pellet mill load, die/roller condition, and feed rate.",
            "Check motor current against rated value and inspect bearings.",
        ],
    },
}


def suggest_fix(machine: str, fault_type: str) -> List[str]:
    if not fault_type or fault_type == "Normal" or fault_type == "N/A":
        return ["No immediate action required. Continue monitoring."]
    lib = FIX_LIBRARY.get(machine, {})
    if fault_type in lib:
        return lib[fault_type]
    ft = fault_type.lower()
    if "oil" in ft:
        return ["Inspect oil system, oil level, filter, and sensor calibration."]
    if "temp" in ft or "overheat" in ft:
        return ["Inspect temperature source, cooling system, and airflow."]
    if "pressure" in ft:
        return ["Inspect pressure source, piping, valve condition, and sensor reading."]
    return ["Perform targeted inspection based on OEM maintenance manual."]


# =========================
# Sequence / lag features
# =========================
def add_history_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    history_steps: int = 3,
) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        s = clean_numeric_series(out[col])
        if s.notna().sum() == 0:
            continue

        out[col] = s
        for lag in range(1, history_steps + 1):
            out[f"{col}__lag{lag}"] = s.shift(lag)
            out[f"{col}__diff{lag}"] = s - s.shift(lag)

        out[f"{col}__rollmean3"] = s.rolling(3, min_periods=1).mean()
        out[f"{col}__rollstd3"] = s.rolling(3, min_periods=1).std()

    return out


# =========================
# RUL target
# =========================
def build_rul_target(df: pd.DataFrame, machine: str, time_key: str, cond_col: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    if time_key in out.columns:
        out[time_key] = pd.to_numeric(out[time_key], errors="coerce").ffill()
    else:
        out[time_key] = np.arange(len(out), dtype=float)

    out[cond_col] = out[cond_col].apply(normalize_condition_text)

    if machine == "pellet":
        hours_col = find_matching_column(out, "HOURS")
        if hours_col is not None:
            hrs = clean_numeric_series(out[hours_col]).fillna(0.0)
        else:
            hrs = pd.Series(np.ones(len(out)), index=out.index)
    elif "OPERATING HOURS" in out.columns:
        hrs = clean_numeric_series(out["OPERATING HOURS"]).fillna(0.0)
    else:
        hrs = pd.Series(np.ones(len(out)), index=out.index)

    out["_hrs"] = hrs.clip(lower=0.0)
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
    out.loc[out["_is_fault"] == 1, "rul_hours"] = 0.0
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

    if cond_col not in df.columns:
        match = find_matching_column(df, cond_col)
        if match is None:
            raise KeyError(f"Condition column '{cond_col}' not found for machine '{machine}'.")
        cond_col = match
        cfg["condition_col"] = cond_col

    if cfg["time_key"] not in df.columns:
        match = find_matching_column(df, cfg["time_key"])
        if match is not None:
            cfg["time_key"] = match

    df[cond_col] = df[cond_col].apply(normalize_condition_text)
    df["condition_severity"] = df[cond_col].apply(map_condition_to_severity)
    df["fault_type"] = df[cond_col].apply(extract_fault_type)
    df["fault_flag"] = df["fault_type"].apply(fault_flag_from_type)

    protected_cols = {cond_col, "condition_severity", "fault_type", "fault_flag"}
    for c in df.columns:
        if c in protected_cols:
            continue

        if machine == "pellet" and normalize_col_name(c) == "FEED TYPE":
            df[c] = df[c].astype(str).replace({"nan": np.nan}).str.strip()
            continue

        s_num = clean_numeric_series(df[c])
        if s_num.notna().mean() >= 0.70:
            df[c] = s_num

    if machine == "pellet":
        df = clean_pellet_rows(df, cond_col)

    return df.reset_index(drop=True)


def build_modeling_frame(machine: str, history_steps: int = 3) -> pd.DataFrame:
    df = load_machine_df(machine)
    cfg = MACHINES[machine]
    cond_col = cfg["condition_col"]
    time_key = cfg["time_key"]

    reserved = {cond_col, time_key, "condition_severity", "fault_type", "fault_flag"}
    numeric_candidate_cols = []

    for c in df.columns:
        if c in reserved:
            continue

        if machine == "pellet" and normalize_col_name(c) == "FEED TYPE":
            continue

        if machine == "pellet" and normalize_col_name(c) == "OPERATING HOURS":
            continue

        s_num = clean_numeric_series(df[c])
        if s_num.notna().sum() > 0:
            df[c] = s_num
            numeric_candidate_cols.append(c)

    if cfg["use_history"]:
        df = add_history_features(df, numeric_candidate_cols, history_steps=history_steps)

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
                f"Please update MODEL_FILE_IDS in maintenance_ml.py."
            )
        _download_from_gdrive(file_id, output_path)

    return output_path


def horizon_artifact_path(machine: str) -> str:
    return os.path.join(MODEL_DIR, HORIZON_MODEL_FILES[machine])


def save_horizon_bundle(machine: str, bundle: Dict[str, Any]) -> None:
    joblib.dump(bundle, horizon_artifact_path(machine))


def ensure_horizon_bundle_exists(machine: str) -> str:
    if machine not in HORIZON_MODEL_FILES:
        raise ValueError(f"No horizon model configured for machine: {machine}")

    if machine not in HORIZON_MODEL_FILE_IDS:
        raise ValueError(f"No Google Drive horizon file ID configured for machine: {machine}")

    output_path = horizon_artifact_path(machine)

    if not os.path.exists(output_path):
        file_id = HORIZON_MODEL_FILE_IDS[machine]
        if not file_id or file_id.startswith("PASTE_"):
            raise ValueError(
                f"Google Drive file ID for horizon model '{machine}' is not set. "
                f"Please update HORIZON_MODEL_FILE_IDS in maintenance_ml.py."
            )
        _download_from_gdrive(file_id, output_path)

    return output_path


@st.cache_resource
def load_bundle(machine: str) -> Dict[str, Any]:
    p = ensure_bundle_exists(machine)
    if not os.path.exists(p):
        return {}
    return joblib.load(p)


@st.cache_resource
def load_horizon_bundle(machine: str) -> Dict[str, Any]:
    p = ensure_horizon_bundle_exists(machine)
    if not os.path.exists(p):
        return {}
    return joblib.load(p)


# =========================
# Inference helpers
# =========================
def get_latest_feature_columns(bundle: Dict[str, Any]) -> List[str]:
    return bundle.get("feature_columns", [])


def base_feature_from_history_feature(col: str) -> str:
    return re.sub(r"__(lag|diff)\d+$", "", str(col)).replace("__rollmean3", "").replace("__rollstd3", "")


def build_history_input_frame(
    machine: str,
    history_rows: List[Dict[str, Any]],
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    df = pd.DataFrame(history_rows).copy()

    cfg = MACHINES[machine]
    cond_col = cfg["condition_col"]
    time_key = cfg["time_key"]

    if cond_col not in df.columns:
        df[cond_col] = "Normal"
    if time_key not in df.columns:
        df[time_key] = list(range(1, len(df) + 1))

    if "condition_severity" not in df.columns:
        df["condition_severity"] = "Normal"
    if "fault_type" not in df.columns:
        df["fault_type"] = "Normal"
    if "fault_flag" not in df.columns:
        df["fault_flag"] = "Normal"

    for c in df.columns:
        if c not in [cond_col, time_key, "condition_severity", "fault_type", "fault_flag"]:
            df[c] = clean_numeric_series(df[c])

    reserved = {cond_col, time_key, "condition_severity", "fault_type", "fault_flag"}
    numeric_candidate_cols = [c for c in df.columns if c not in reserved]

    if MACHINES[machine]["use_history"]:
        df = add_history_features(df, numeric_candidate_cols, history_steps=3)

    X = df.iloc[[-1]].copy()
    X = X.drop(
        columns=[c for c in [cond_col, time_key, "condition_severity", "fault_type", "fault_flag"] if c in X.columns],
        errors="ignore"
    )

    trained_cols = bundle["feature_columns"]
    for c in trained_cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[trained_cols].copy()
    return X


def infer_genset_fault_type_rules(row_dict: Dict[str, Any]) -> str:
    row = pd.Series(row_dict)

    def get(pattern: str) -> float:
        for c in row.index:
            if pattern in normalize_col_name(c):
                return pd.to_numeric(row[c], errors="coerce")
        return np.nan

    oil = get("OIL PRESSURE")
    freq = get("FREQUENCY")
    cool = get("COOLANT TEMPERATURE")
    v1 = get("VOLTAGE")
    c1 = get("CURRENT")

    if np.isfinite(cool) and cool >= 95:
        return "Critical Overheat"
    if np.isfinite(cool) and cool >= 90:
        return "Overheating"
    if np.isfinite(oil) and oil < 10:
        return "Low Oil Pressure"
    if np.isfinite(oil) and oil > 60:
        return "High Oil Pressure"
    if np.isfinite(freq) and (freq < 58.5 or freq > 61.5):
        return "Frequency Instability"
    if np.isfinite(v1) and v1 < 200:
        return "Voltage Drop"
    if np.isfinite(c1) and c1 > 0 and c1 >= 1.20 * c1:
        return "Current Unbalance"

    return "N/A"


def infer_boiler_fault_type_rules(history_rows: List[Dict[str, Any]]) -> str:
    if not history_rows:
        return "N/A"

    df = pd.DataFrame(history_rows).copy()

    def find_col(pattern: str) -> Optional[str]:
        for c in df.columns:
            if pattern in normalize_col_name(c):
                return c
        return None

    steam_col = find_col("MAIN STREAM PRESSURE")
    fuel_temp_col = find_col("FUEL GAS")
    fuel_press_col = find_col("FUEL PRESSURE")

    water_cols = [c for c in df.columns if "WATER" in normalize_col_name(c)]
    boiler_water_col = water_cols[0] if len(water_cols) >= 1 else None
    fw_tank_col = water_cols[1] if len(water_cols) >= 2 else None

    def delta(col: Optional[str], tail_n: int = 4) -> float:
        if col is None or col not in df.columns:
            return np.nan
        s = clean_numeric_series(df[col]).dropna()
        if len(s) < 2:
            return np.nan
        s = s.tail(tail_n)
        return float(s.iloc[-1] - s.iloc[0])

    steam_d = delta(steam_col)
    fuel_temp_d = delta(fuel_temp_col)
    fuel_press_d = delta(fuel_press_col)
    boiler_water_d = delta(boiler_water_col)
    fw_tank_d = delta(fw_tank_col)

    last = df.iloc[-1]
    steam_now = pd.to_numeric(last.get(steam_col, np.nan), errors="coerce")
    fuel_temp_now = pd.to_numeric(last.get(fuel_temp_col, np.nan), errors="coerce")
    fuel_press_now = pd.to_numeric(last.get(fuel_press_col, np.nan), errors="coerce")

    if np.isfinite(fuel_temp_d) and np.isfinite(steam_d):
        if fuel_temp_d > 35 and steam_d < -5:
            return "High Flue Gas Temp Trip"

    if np.isfinite(fuel_press_d) and fuel_press_d < -0.25 and np.isfinite(steam_d) and steam_d < -8:
        return "Burner Trip - Low Fuel Pressure"

    if np.isfinite(fuel_press_d) and fuel_press_d < -0.25 and np.isfinite(fuel_temp_d) and fuel_temp_d < -8:
        return "Fuel Pump Fail"

    if np.isfinite(fuel_press_d) and fuel_press_d > 0.30 and np.isfinite(fuel_temp_d) and fuel_temp_d > 30:
        return "Burner Nozzle Fault"

    if np.isfinite(fw_tank_d) and np.isfinite(boiler_water_d):
        if fw_tank_d < -0.20 and boiler_water_d < -0.08:
            fw_now = pd.to_numeric(last.get(fw_tank_col, np.nan), errors="coerce")
            if np.isfinite(fw_now) and fw_now <= 0.35:
                return "Low Water Trip"
            return "Makeup Valve Stuck"

    if np.isfinite(boiler_water_d) and np.isfinite(steam_d):
        if boiler_water_d > 0.15 and steam_d < -8:
            return "Level Controller Malfunction"

    if np.isfinite(fuel_temp_d) and fuel_temp_d > 20 and np.isfinite(steam_d) and steam_d < -2:
        return "Tube Scaling"

    if np.isfinite(steam_d) and steam_d < -10:
        return "Main Steam Line Leak"

    if np.isfinite(fuel_temp_now) and fuel_temp_now >= 240:
        return "High Flue Gas Temp Trip"

    if np.isfinite(fuel_press_now) and fuel_press_now <= 0.65:
        return "Burner Trip - Low Fuel Pressure"

    if np.isfinite(steam_now) and steam_now <= 80:
        return "Main Steam Line Leak"

    return "N/A"
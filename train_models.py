from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from maintenance_ml import (
    MACHINES,
    load_machine_df,
    build_modeling_frame,
    build_preprocessor,
    save_bundle,
    build_rul_target,
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_col_name(col_name: str) -> str:
    name = str(col_name).strip().upper()
    name = re.sub(r"__\d+$", "", name)
    return name


def is_time_like_column(col_name: str) -> bool:
    name = normalize_col_name(col_name)
    if name in {
        "DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME",
        "DAILY CONDITION", "WEEKLY CONDITION"
    }:
        return True
    prefixes = (
        "DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME",
        "DAILY CONDITION", "WEEKLY CONDITION"
    )
    return any(name.startswith(prefix) for prefix in prefixes)


def should_exclude_feature(machine: str, col_name: str) -> bool:
    name = normalize_col_name(col_name)

    if is_time_like_column(col_name):
        return True

    if "CONDITION" in name:
        return True

    if machine == "genset" and name == "REMARKS":
        return True

    if machine == "pellet" and name in {"OPERATING HOURS", "HOURS", "FEED TYPE"}:
        return True

    if machine == "boiler" and name == "OPERATING HOURS":
        return True

    return False


def filter_feature_columns(machine: str, X: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in X.columns if should_exclude_feature(machine, c)]
    X = X.drop(columns=cols_to_drop, errors="ignore")
    X = X.dropna(axis=1, how="all")

    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=constant_cols, errors="ignore")
    return X


def get_classifiers() -> Dict[str, Any]:
    models = {
        "LogReg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            C=1.0,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "SVM_RBF": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass

    ensemble_estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
        ("gb", GradientBoostingClassifier(random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ]
    models["Ensemble_Voting"] = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
    )

    return models


def get_regressors() -> Dict[str, Any]:
    regs = {
        "RF_Reg": RandomForestRegressor(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GB_Reg": GradientBoostingRegressor(random_state=42),
        "ExtraTrees_Reg": ExtraTreesRegressor(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "AdaBoost_Reg": AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
        ),
        "HistGB_Reg": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            random_state=42,
        ),
        "SVR_RBF": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "KNN_Reg": KNeighborsRegressor(n_neighbors=5),
    }

    try:
        from xgboost import XGBRegressor
        regs["XGB_Reg"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        regs["LightGBM_Reg"] = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass

    ensemble_regs = [
        (
            "rf",
            RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
        ("gb", GradientBoostingRegressor(random_state=42)),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=400,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ]
    regs["Ensemble_Voting_Reg"] = VotingRegressor(estimators=ensemble_regs)

    return regs


def safe_stratified_cv(y: pd.Series):
    min_class_count = y.value_counts().min()
    n_splits = min(5, int(min_class_count)) if pd.notna(min_class_count) else 0
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def safe_kfold(n_rows: int):
    n_splits = min(5, max(2, n_rows // 10))
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate_classifier(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    cv = safe_stratified_cv(y)
    if cv is None:
        raise ValueError(f"Not enough samples per class for {name}")

    f1_macro = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", error_score="raise").mean()
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", error_score="raise").mean()

    y_pred_cv = cross_val_predict(pipe, X, y, cv=cv)
    labels = sorted(pd.Series(y).astype(str).unique().tolist())
    cm = confusion_matrix(
        y.astype(str),
        pd.Series(y_pred_cv).astype(str),
        labels=labels,
    )

    report = classification_report(
        y.astype(str),
        pd.Series(y_pred_cv).astype(str),
        output_dict=True,
        zero_division=0,
    )

    pipe.fit(X, y)

    return {
        "model": name,
        "f1_macro": float(f1_macro),
        "accuracy": float(acc),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fitted_pipeline": pipe,
    }


def evaluate_regressor(name: str, pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    cv = safe_kfold(len(X))
    preds = cross_val_predict(pipe, X, y, cv=cv)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    pipe.fit(X, y)

    return {
        "model": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "y_true": y.tolist(),
        "y_pred": preds.tolist(),
        "fitted_pipeline": pipe,
    }


def plot_bar_ranking(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    save_path: str,
    ascending: bool = False,
) -> None:
    dff = df.sort_values(y_col, ascending=ascending).copy()

    plt.figure(figsize=(11, 6))
    bars = plt.bar(dff[x_col], dff[y_col])
    plt.title(title, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, save_path: str) -> None:
    plt.figure(figsize=(7.2, 6.2))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.title(title, fontweight="bold", fontsize=13)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() * 0.55 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                f"{value:d}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if value > thresh else "black",
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_classification_report_table(report_dict: dict, save_path: str) -> None:
    pd.DataFrame(report_dict).T.to_csv(save_path, index=True)


def get_feature_importance_from_pipeline(
    pipe: Pipeline,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        pre = pipe.named_steps["pre"]
        model = pipe.named_steps["model"]

        feature_names = pre.get_feature_names_out()
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                importances = np.mean(np.abs(coef), axis=0)

        if importances is not None and len(importances) == len(feature_names):
            return feature_names, importances
    except Exception:
        pass

    return None, None


def plot_feature_importance(
    feature_names: np.ndarray,
    importances: np.ndarray,
    title: str,
    save_path: str,
    top_n: int = 15,
) -> None:
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title(title, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.title(title, fontweight="bold")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def get_excel_source_path() -> str:
    env_path = os.environ.get("QPLC_XLSX_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    base_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        base_dir / "QPLC-Maintenance Data.xlsx",
        cwd / "QPLC-Maintenance Data.xlsx",
        base_dir / "data" / "QPLC-Maintenance Data.xlsx",
        cwd / "data" / "QPLC-Maintenance Data.xlsx",
        cwd / "qplc_ai_app" / "data" / "QPLC-Maintenance Data.xlsx",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find QPLC-Maintenance Data.xlsx. "
        "Expected it in qplc_ai_app/data or beside train_models.py."
    )


def is_fault_text(v: Any) -> bool:
    s = str(v).strip().lower()

    if s in {"", "-", "nan", "none", "normal", "ok", "good", "healthy", "no fault"}:
        return False

    abnormal_keywords = [
        "fault",
        "trend",
        "repair",
        "trip",
        "leak",
        "low",
        "high",
        "overload",
        "overheat",
        "failure",
        "malfunction",
        "unstable",
        "critical",
        "reset",
    ]
    return any(k in s for k in abnormal_keywords)


def read_raw_machine_sheet(machine: str) -> pd.DataFrame:
    source_path = get_excel_source_path()
    print(f"[LSTM DEBUG] Using Excel file: {source_path}")

    sheet_map = {
        "boiler": "Boiler",
        "genset": "Gen Set",
        "pellet": "Pellet Mill",
    }
    sheet_name = sheet_map[machine]

    df = pd.read_excel(source_path, sheet_name=sheet_name, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    rename_common = {
        "DAILY CONDITION": "DAILY_CONDITION",
    }
    df = df.rename(columns=rename_common)

    if "DAY" not in df.columns:
        raise ValueError(f"'DAY' column not found in sheet: {sheet_name}. Columns found: {list(df.columns)}")

    df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce").ffill()
    df = df.dropna(subset=["DAY"]).copy()
    df["DAY"] = df["DAY"].astype(int)

    print(f"[LSTM DEBUG] {machine} raw columns: {list(df.columns)}")
    return df


def aggregate_daily_for_lstm(machine: str) -> pd.DataFrame:
    df = read_raw_machine_sheet(machine).copy()

    if machine == "boiler":
        rename_map = {
            "DAY": "DAY",
            "BOILER WATER": "BOILER_WATER",
            "BOILER MAIN STEAM PRESSURE": "MAIN_STEAM_PRESSURE",
            "BOILER FUEL GAS": "FUEL_GAS_TEMP",
            "BURNER FUEL PRESSURE": "BURNER_FUEL_PRESSURE",
            "FEED WATER TANK": "FEED_WATER_TANK_LEVEL",
            "FEED WATER TEMP": "FEED_WATER_TEMP",
            "DAILY_CONDITION": "DAILY_CONDITION",
        }
        df = df.rename(columns=rename_map)

        numeric_cols = [
            "BOILER_WATER",
            "MAIN_STEAM_PRESSURE",
            "FUEL_GAS_TEMP",
            "BURNER_FUEL_PRESSURE",
            "FEED_WATER_TANK_LEVEL",
            "FEED_WATER_TEMP",
        ]

    elif machine == "pellet":
        rename_map = {
            "DAY": "DAY",
            "AMP1 (100AMP)": "AMP1",
            "AMP2 (100AMP)": "AMP2",
            "US PRESS (4.5-8BARS)": "US_PRESS",
            "DS PRESS (1.8-3BARS)": "DS_PRESS",
            "FEEDER RATE (60HERTZ) - (%)": "FEEDER_RATE",
            "TEMPERATURE": "TEMPERATURE",
            "DAILY_CONDITION": "DAILY_CONDITION",
        }
        df = df.rename(columns=rename_map)

        numeric_cols = [
            "AMP1",
            "AMP2",
            "US_PRESS",
            "DS_PRESS",
            "FEEDER_RATE",
            "TEMPERATURE",
        ]

    else:
        raise ValueError(f"Unsupported machine for LSTM daily aggregation: {machine}")

    required_cols = ["DAY", "DAILY_CONDITION"] + numeric_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns for {machine}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["fault_row"] = df["DAILY_CONDITION"].apply(is_fault_text).astype(int)

    agg_map = {}
    for c in numeric_cols:
        agg_map[f"{c}_mean"] = (c, "mean")
        agg_map[f"{c}_min"] = (c, "min")
        agg_map[f"{c}_max"] = (c, "max")
        agg_map[f"{c}_std"] = (c, "std")
        agg_map[f"{c}_last"] = (c, lambda x: x.dropna().iloc[-1] if len(x.dropna()) else np.nan)

    daily = df.groupby("DAY").agg(
        **agg_map,
        fault_flag_binary=("fault_row", "max"),
    ).reset_index()

    daily = daily.sort_values("DAY").reset_index(drop=True)

    std_cols = [c for c in daily.columns if c.endswith("_std")]
    for c in std_cols:
        daily[c] = daily[c].fillna(0)

    print(f"[LSTM DEBUG] {machine} daily columns: {list(daily.columns)}")
    return daily


def build_days_to_fault_target(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy().reset_index(drop=True)

    fault_indices = np.where(df["fault_flag_binary"].values == 1)[0]
    days_to_fault = np.full(len(df), np.nan, dtype=float)

    if len(fault_indices) == 0:
        df["days_to_fault"] = days_to_fault
        return df

    for i in range(len(df)):
        next_faults = fault_indices[fault_indices >= i]
        if len(next_faults) == 0:
            continue
        next_idx = next_faults[0]
        days_to_fault[i] = float(df.loc[next_idx, "DAY"] - df.loc[i, "DAY"])

    df["days_to_fault"] = days_to_fault
    return df


def make_lstm_sequences(
    df_daily: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 7,
) -> tuple[np.ndarray, np.ndarray, list[int], StandardScaler]:
    df = df_daily.copy().reset_index(drop=True)

    valid = df["days_to_fault"].notna()
    df = df.loc[valid].reset_index(drop=True)

    if len(df) <= seq_len:
        raise ValueError("Not enough daily rows to build LSTM sequences.")

    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[feature_cols].astype(float).values)

    X_seq, y_seq, day_refs = [], [], []

    for i in range(seq_len, len(df)):
        X_seq.append(X_all[i - seq_len:i, :])
        y_seq.append(float(df.loc[i, "days_to_fault"]))
        day_refs.append(int(df.loc[i, "DAY"]))

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), day_refs, scaler


def build_lstm_regressor(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="relu"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_lstm_days_to_fault(
    machine: str,
    results_dir: str,
    seq_len: int = 7,
    epochs: int = 100,
    batch_size: int = 16,
) -> Optional[Dict[str, Any]]:
    if not TF_AVAILABLE:
        print(f"[WARNING] TensorFlow is not installed. Skipping LSTM for {machine}.")
        return None

    if machine not in {"boiler", "pellet"}:
        return None

    daily = aggregate_daily_for_lstm(machine)
    daily = build_days_to_fault_target(daily)

    feature_cols = [
        c for c in daily.columns
        if c not in {"DAY", "fault_flag_binary", "days_to_fault"}
        and pd.api.types.is_numeric_dtype(daily[c])
    ]

    usable = int(daily["days_to_fault"].notna().sum())
    fault_days = int(daily["fault_flag_binary"].sum())

    print(f"[LSTM DEBUG] {machine} unique days = {len(daily)}")
    print(f"[LSTM DEBUG] {machine} fault days = {fault_days}")
    print(f"[LSTM DEBUG] {machine} usable days_to_fault rows = {usable}")
    print(f"[LSTM DEBUG] {machine} feature count = {len(feature_cols)}")

    if len(feature_cols) == 0:
        print(f"[WARNING] No valid numeric daily features for LSTM on {machine}.")
        return None

    if usable < (seq_len + 10):
        print(f"[WARNING] Not enough valid daily data for LSTM on {machine}. Need more rows.")
        return None

    X, y, day_refs, scaler = make_lstm_sequences(daily, feature_cols, seq_len=seq_len)

    split_idx = int(len(X) * 0.8)
    if split_idx < 1 or len(X) - split_idx < 1:
        print(f"[WARNING] Not enough sequences after split for {machine}.")
        return None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    day_test = day_refs[split_idx:]

    model = build_lstm_regressor((X.shape[1], X.shape[2]))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        shuffle=False,
    )

    pred_test = model.predict(X_test, verbose=0).reshape(-1)
    pred_test = np.maximum(pred_test, 0)

    mae = float(mean_absolute_error(y_test, pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    r2 = float(r2_score(y_test, pred_test))

    machine_dir = os.path.join(results_dir, machine)
    ensure_dir(machine_dir)

    model_path = os.path.join(machine_dir, f"{machine}_lstm_days_to_fault.keras")
    scaler_path = os.path.join(machine_dir, f"{machine}_lstm_scaler.npz")
    meta_path = os.path.join(machine_dir, f"{machine}_lstm_metadata.json")
    preds_path = os.path.join(machine_dir, f"{machine}_lstm_test_predictions.csv")
    loss_plot_path = os.path.join(machine_dir, f"{machine}_lstm_training_loss.png")
    avp_plot_path = os.path.join(machine_dir, f"{machine}_lstm_actual_vs_predicted_days.png")
    residual_plot_path = os.path.join(machine_dir, f"{machine}_lstm_residuals_days.png")

    model.save(model_path)

    np.savez(
        scaler_path,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "machine": machine,
                "seq_len": seq_len,
                "feature_cols": feature_cols,
                "target": "days_to_fault",
                "metrics": {
                    "mae_days": mae,
                    "rmse_days": rmse,
                    "r2": r2,
                },
                "debug": {
                    "unique_days": int(len(daily)),
                    "fault_days": fault_days,
                    "usable_days_to_fault_rows": usable,
                },
            },
            f,
            indent=2,
        )

    pd.DataFrame({
        "DAY": day_test,
        "actual_days_to_fault": y_test,
        "predicted_days_to_fault": pred_test,
    }).to_csv(preds_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{machine.upper()} - LSTM Training Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=pred_test,
        title=f"{machine.upper()} - LSTM Actual vs Predicted Days to Fault",
        save_path=avp_plot_path,
    )

    plot_residuals(
        y_true=y_test,
        y_pred=pred_test,
        title=f"{machine.upper()} - LSTM Residual Plot (Days to Fault)",
        save_path=residual_plot_path,
    )

    return {
        "machine": machine,
        "model_type": "LSTM_days_to_fault",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metadata_path": meta_path,
        "predictions_path": preds_path,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "mae_days": mae,
        "rmse_days": rmse,
        "r2": r2,
    }


def save_results(bundle: Dict[str, Any], machine: str, results_dir: str) -> None:
    ensure_dir(results_dir)
    machine_dir = os.path.join(results_dir, machine)
    ensure_dir(machine_dir)

    sections = ["severity", "fault_flag", "fault_type"]

    for sec in sections:
        leader = bundle[sec]["leaderboard"]
        if not leader:
            continue

        rows = []
        for item in leader:
            rows.append({
                "model": item["model"],
                "f1_macro": item["f1_macro"],
                "accuracy": item["accuracy"],
            })

        df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(machine_dir, f"{machine}_{sec}_leaderboard.csv"), index=False)

        plot_bar_ranking(
            df,
            x_col="model",
            y_col="f1_macro",
            title=f"{machine.upper()} - {sec.replace('_', ' ').title()} Model Comparison",
            ylabel="F1 Macro",
            save_path=os.path.join(machine_dir, f"{machine}_{sec}_f1_ranking.png"),
            ascending=False,
        )

        best_model_name = bundle[sec]["best_model"]
        best_item = next((x for x in leader if x["model"] == best_model_name), None)

        if best_item is not None:
            labels = best_item["labels"]
            cm = np.array(best_item["confusion_matrix"])

            plot_confusion_matrix(
                cm,
                labels,
                title=f"{machine.upper()} - {sec.replace('_', ' ').title()} Confusion Matrix ({best_model_name})",
                save_path=os.path.join(machine_dir, f"{machine}_{sec}_confusion_matrix.png"),
            )

            save_classification_report_table(
                best_item["classification_report"],
                os.path.join(machine_dir, f"{machine}_{sec}_classification_report.csv"),
            )

            pipe = best_item.get("fitted_pipeline")
            if pipe is not None:
                feature_names, importances = get_feature_importance_from_pipeline(pipe)
                if feature_names is not None and importances is not None:
                    plot_feature_importance(
                        feature_names,
                        importances,
                        title=f"{machine.upper()} - {sec.replace('_', ' ').title()} Feature Importance ({best_model_name})",
                        save_path=os.path.join(machine_dir, f"{machine}_{sec}_feature_importance.png"),
                        top_n=15,
                    )

    if bundle["rul"] and bundle["rul"]["leaderboard"]:
        leader = bundle["rul"]["leaderboard"]
        rows = []
        for item in leader:
            rows.append({
                "model": item["model"],
                "mae": item["mae"],
                "rmse": item["rmse"],
                "r2": item["r2"],
            })

        df = pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.to_csv(os.path.join(machine_dir, f"{machine}_rul_leaderboard.csv"), index=False)

        plot_bar_ranking(
            df,
            x_col="model",
            y_col="mae",
            title=f"{machine.upper()} - RUL Model Comparison",
            ylabel="MAE (Hours)",
            save_path=os.path.join(machine_dir, f"{machine}_rul_mae_ranking.png"),
            ascending=True,
        )

        best_model_name = bundle["rul"]["best_model"]
        best_item = next((x for x in leader if x["model"] == best_model_name), None)

        if best_item is not None:
            y_true = np.array(best_item["y_true"])
            y_pred = np.array(best_item["y_pred"])

            plot_actual_vs_predicted(
                y_true,
                y_pred,
                title=f"{machine.upper()} - Actual vs Predicted RUL ({best_model_name})",
                save_path=os.path.join(machine_dir, f"{machine}_rul_actual_vs_predicted.png"),
            )

            plot_residuals(
                y_true,
                y_pred,
                title=f"{machine.upper()} - RUL Residual Plot ({best_model_name})",
                save_path=os.path.join(machine_dir, f"{machine}_rul_residuals.png"),
            )

            pipe = best_item.get("fitted_pipeline")
            if pipe is not None:
                feature_names, importances = get_feature_importance_from_pipeline(pipe)
                if feature_names is not None and importances is not None:
                    plot_feature_importance(
                        feature_names,
                        importances,
                        title=f"{machine.upper()} - RUL Feature Importance ({best_model_name})",
                        save_path=os.path.join(machine_dir, f"{machine}_rul_feature_importance.png"),
                        top_n=15,
                    )


def save_lstm_summary(all_lstm_rows: list[dict], results_dir: str) -> None:
    if not all_lstm_rows:
        return
    df = pd.DataFrame(all_lstm_rows)
    df.to_csv(os.path.join(results_dir, "deep_learning_lstm_summary.csv"), index=False)


def train_machine(machine: str) -> Dict[str, Any]:
    cfg = MACHINES[machine]
    df_raw = load_machine_df(machine)
    df = build_modeling_frame(machine, history_steps=3)

    cond_col = cfg["condition_col"]
    time_key = cfg["time_key"]

    drop_cols = {cond_col, "condition_severity", "fault_type", "fault_flag"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = filter_feature_columns(machine, X)

    valid_mask = ~X.isna().all(axis=1)
    X = X.loc[valid_mask].copy()

    y_sev = df.loc[valid_mask, "condition_severity"].astype(str)
    y_fault_flag = df.loc[valid_mask, "fault_flag"].astype(str)

    pre, _, _ = build_preprocessor(X)
    clf_models = get_classifiers()

    sev_results = []
    sev_pipes = {}
    for name, model in clf_models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        try:
            r = evaluate_classifier(name, pipe, X, y_sev)
            sev_results.append(r)
            sev_pipes[name] = pipe
        except Exception as e:
            print(f"[WARNING] Skipping severity model '{name}' for {machine}: {e}")

    if not sev_results:
        raise RuntimeError(f"No severity models trained successfully for machine: {machine}")
    sev_leader = sorted(sev_results, key=lambda d: d["f1_macro"], reverse=True)

    ff_results = []
    ff_pipes = {}
    for name, model in clf_models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        try:
            r = evaluate_classifier(name, pipe, X, y_fault_flag)
            ff_results.append(r)
            ff_pipes[name] = pipe
        except Exception as e:
            print(f"[WARNING] Skipping fault flag model '{name}' for {machine}: {e}")

    if not ff_results:
        raise RuntimeError(f"No fault-flag models trained successfully for machine: {machine}")
    ff_leader = sorted(ff_results, key=lambda d: d["f1_macro"], reverse=True)

    ft_leader = []
    ft_pipes = {}
    ft_best = None

    if cfg["fault_type_mode"] == "ml":
        fault_rows = df.loc[valid_mask, "fault_type"].astype(str) != "Normal"
        X_fault = X.loc[fault_rows].copy()
        y_fault_type = df.loc[valid_mask, "fault_type"].astype(str).loc[fault_rows]

        if len(X_fault) >= 20 and y_fault_type.nunique() >= 2:
            pre_fault, _, _ = build_preprocessor(X_fault)
            for name, model in clf_models.items():
                pipe = Pipeline([("pre", pre_fault), ("model", model)])
                try:
                    r = evaluate_classifier(name, pipe, X_fault, y_fault_type)
                    ft_leader.append(r)
                    ft_pipes[name] = pipe
                except Exception as e:
                    print(f"[WARNING] Skipping fault type model '{name}' for {machine}: {e}")

            ft_leader = sorted(ft_leader, key=lambda d: d["f1_macro"], reverse=True)
            if ft_leader:
                ft_best = ft_leader[0]["model"]
                ft_pipes[ft_best].fit(X_fault, y_fault_type)

    df_rul = build_rul_target(df_raw, machine, time_key, cond_col)
    if cfg["use_history"]:
        df_rul = build_modeling_frame(machine, history_steps=3).join(
            df_rul[["rul_hours"]], how="left"
        )

    rul_bundle = None
    rul_leader = []

    if cfg["rul_enabled"] and "rul_hours" in df_rul.columns and df_rul["rul_hours"].notna().sum() >= 20:
        Xr = df_rul.drop(
            columns=[cond_col, "condition_severity", "fault_type", "fault_flag", "rul_hours"],
            errors="ignore",
        )
        Xr = filter_feature_columns(machine, Xr)
        valid_rul = df_rul["rul_hours"].notna() & ~Xr.isna().all(axis=1)
        Xr = Xr.loc[valid_rul].copy()
        yr = df_rul.loc[valid_rul, "rul_hours"].values.astype(float)

        pre_r, _, _ = build_preprocessor(Xr)
        regs = get_regressors()
        reg_pipes = {}

        for name, reg in regs.items():
            pipe = Pipeline([("pre", pre_r), ("model", reg)])
            try:
                r = evaluate_regressor(name, pipe, Xr, yr)
                rul_leader.append(r)
                reg_pipes[name] = pipe
            except Exception as e:
                print(f"[WARNING] Skipping RUL model '{name}' for {machine}: {e}")

        if rul_leader:
            rul_leader = sorted(rul_leader, key=lambda d: d["mae"])
            best_rul = rul_leader[0]["model"]
            reg_pipes[best_rul].fit(Xr, yr)
            rul_bundle = {
                "best_model": best_rul,
                "pipelines": reg_pipes,
                "leaderboard": rul_leader,
            }

    best_sev = sev_leader[0]["model"]
    best_ff = ff_leader[0]["model"]

    sev_pipes[best_sev].fit(X, y_sev)
    ff_pipes[best_ff].fit(X, y_fault_flag)

    bundle = {
        "machine": machine,
        "feature_columns": list(X.columns),
        "severity": {
            "best_model": best_sev,
            "pipelines": sev_pipes,
            "leaderboard": sev_leader,
        },
        "fault_flag": {
            "best_model": best_ff,
            "pipelines": ff_pipes,
            "leaderboard": ff_leader,
        },
        "fault_type": {
            "best_model": ft_best,
            "pipelines": ft_pipes,
            "leaderboard": ft_leader,
            "mode": cfg["fault_type_mode"],
        },
        "rul": rul_bundle,
    }

    return bundle


def main() -> None:
    results_dir = "results"
    ensure_dir(results_dir)

    summary_rows = []
    lstm_rows = []

    for m in MACHINES.keys():
        print(f"Training: {m}")
        bundle = train_machine(m)
        save_bundle(m, bundle)

        print(f"Saved bundle for: {m}")
        print("Best severity:", bundle["severity"]["best_model"])
        print("Best fault flag:", bundle["fault_flag"]["best_model"])
        print("Best fault type:", bundle["fault_type"]["best_model"] if bundle["fault_type"]["best_model"] else "rule_based")
        print("Best RUL:", bundle["rul"]["best_model"] if bundle["rul"] else "N/A")
        print("Final feature columns used:")
        print(bundle["feature_columns"])
        print("-" * 50)

        save_results(bundle, m, results_dir)

        summary_rows.append({
            "machine": m,
            "best_severity_model": bundle["severity"]["best_model"],
            "best_fault_flag_model": bundle["fault_flag"]["best_model"],
            "best_fault_type_model": bundle["fault_type"]["best_model"] if bundle["fault_type"]["best_model"] else "rule_based",
            "best_rul_model": bundle["rul"]["best_model"] if bundle["rul"] else None,
        })

        if m in {"boiler", "pellet"}:
            try:
                lstm_result = train_lstm_days_to_fault(
                    machine=m,
                    results_dir=results_dir,
                    seq_len=7,
                    epochs=100,
                    batch_size=16,
                )
                if lstm_result is not None:
                    lstm_rows.append(lstm_result)
                    print(
                        f"[LSTM] {m} -> "
                        f"MAE(days): {lstm_result['mae_days']:.3f}, "
                        f"RMSE(days): {lstm_result['rmse_days']:.3f}, "
                        f"R2: {lstm_result['r2']:.3f}"
                    )
            except Exception as e:
                print(f"[WARNING] LSTM training failed for {m}: {e}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(results_dir, "overall_best_models_summary.csv"),
        index=False
    )

    save_lstm_summary(lstm_rows, results_dir)


if __name__ == "__main__":
    main()
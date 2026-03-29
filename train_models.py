from __future__ import annotations

import os
import re
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

from maintenance_ml import (
    MACHINES,
    load_machine_df,
    build_modeling_frame,
    build_preprocessor,
    save_bundle,
    save_horizon_bundle,
    build_rul_target,
    find_matching_column,
)

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


def get_horizon_classifiers() -> Dict[str, Any]:
    models = {
        "LogReg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            C=1.0,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=800,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=1,
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
            n_estimators=800,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass

    ensemble_estimators = [
        ("et", ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
    ]
    models["Ensemble_Voting"] = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
    )

    return models


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


def aggregate_daily_for_horizon(machine: str) -> pd.DataFrame:
    df = load_machine_df(machine).copy()
    cfg = MACHINES[machine]

    time_key = cfg["time_key"]
    cond_col = cfg["condition_col"]

    day_col = find_matching_column(df, time_key)
    if day_col is None:
        raise ValueError(f"Could not find time key '{time_key}' for machine '{machine}'")

    cond_col_real = find_matching_column(df, cond_col)
    if cond_col_real is None:
        raise ValueError(f"Could not find condition column '{cond_col}' for machine '{machine}'")

    if machine == "boiler":
        feature_targets = {
            "BOILER_WATER": "BOILER WATER",
            "MAIN_STEAM_PRESSURE": "BOILER MAIN STEAM PRESSURE",
            "FUEL_GAS_TEMP": "BOILER FUEL GAS",
            "BURNER_FUEL_PRESSURE": "BURNER FUEL PRESSURE",
            "FEED_WATER_TANK_LEVEL": "FEED WATER TANK",
            "FEED_WATER_TEMP": "FEED WATER TEMP",
        }
    elif machine == "pellet":
        feature_targets = {
            "AMP1": "AMP1",
            "AMP2": "AMP2",
            "US_PRESS": "US PRESS",
            "DS_PRESS": "DS PRESS",
            "FEEDER_RATE": "FEEDER RATE",
            "TEMPERATURE": "TEMPERATURE",
        }
    else:
        raise ValueError(f"Unsupported machine for horizon aggregation: {machine}")

    rename_map = {day_col: "DAY", cond_col_real: "DAILY_CONDITION"}
    numeric_cols = []

    for new_name, target in feature_targets.items():
        real_col = find_matching_column(df, target)
        if real_col is not None:
            rename_map[real_col] = new_name
            numeric_cols.append(new_name)

    df = df.rename(columns=rename_map).copy()

    required_cols = ["DAY", "DAILY_CONDITION"] + numeric_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {machine}: {missing}")

    df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce").ffill()
    df = df.dropna(subset=["DAY"]).copy()
    df["DAY"] = df["DAY"].astype(int)

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
        daily[c] = daily[c].fillna(0.0)

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


def map_days_to_horizon_class(days: Any) -> Any:
    if pd.isna(days):
        return np.nan
    d = float(days)
    if d <= 10:
        return "Risk"
    return "Safe"


def build_fault_horizon_class_target(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = build_days_to_fault_target(daily_df)
    df["fault_horizon_class"] = df["days_to_fault"].apply(map_days_to_horizon_class)
    return df


def print_horizon_class_distribution(machine: str, y: pd.Series) -> None:
    dist = y.value_counts(dropna=False).to_dict()
    print(f"[HORIZON CLASS DISTRIBUTION] {machine}: {dist}")


def make_horizon_tabular_sequences(
    df_daily: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 7,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    df = df_daily.copy().reset_index(drop=True)
    df = df[df["fault_horizon_class"].notna()].reset_index(drop=True)

    rows = []
    targets = []
    day_refs = []

    if len(df) < seq_len:
        raise ValueError("Not enough daily rows for horizon sequence generation.")

    for i in range(seq_len - 1, len(df)):
        window = df.iloc[i - seq_len + 1:i + 1].copy()
        row = {}

        for step_idx in range(seq_len):
            step = window.iloc[step_idx]
            suffix = f"d{step_idx + 1}"
            for col in feature_cols:
                row[f"{col}__{suffix}"] = step[col]

        current = window.iloc[-1]
        oldest = window.iloc[0]
        for col in feature_cols:
            vals = window[col].astype(float).values
            row[f"{col}__delta"] = float(current[col] - oldest[col])
            row[f"{col}__wmean"] = float(np.mean(vals))
            row[f"{col}__wstd"] = float(np.std(vals))

        rows.append(row)
        targets.append(df.loc[i, "fault_horizon_class"])
        day_refs.append(int(df.loc[i, "DAY"]))

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="fault_horizon_class")
    return X, y, day_refs


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


def save_horizon_results(bundle: Dict[str, Any], machine: str, results_dir: str) -> None:
    if not bundle or not bundle.get("leaderboard"):
        return

    machine_dir = os.path.join(results_dir, machine)
    ensure_dir(machine_dir)

    rows = []
    for item in bundle["leaderboard"]:
        rows.append({
            "model": item["model"],
            "f1_macro": item["f1_macro"],
            "accuracy": item["accuracy"],
        })

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(os.path.join(machine_dir, f"{machine}_fault_horizon_leaderboard.csv"), index=False)

    plot_bar_ranking(
        df,
        x_col="model",
        y_col="f1_macro",
        title=f"{machine.upper()} - Fault Horizon Classifier Comparison",
        ylabel="F1 Macro",
        save_path=os.path.join(machine_dir, f"{machine}_fault_horizon_f1_ranking.png"),
        ascending=False,
    )

    best_name = bundle["best_model"]
    best_item = next((x for x in bundle["leaderboard"] if x["model"] == best_name), None)
    if best_item is not None:
        labels = best_item["labels"]
        cm = np.array(best_item["confusion_matrix"])

        plot_confusion_matrix(
            cm,
            labels,
            title=f"{machine.upper()} - Fault Horizon Confusion Matrix ({best_name})",
            save_path=os.path.join(machine_dir, f"{machine}_fault_horizon_confusion_matrix.png"),
        )

        save_classification_report_table(
            best_item["classification_report"],
            os.path.join(machine_dir, f"{machine}_fault_horizon_classification_report.csv"),
        )


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


def train_fault_horizon_classifier(
    machine: str,
    seq_len: int = 3,
) -> Optional[Dict[str, Any]]:
    if machine not in {"boiler", "pellet"}:
        return None

    daily = aggregate_daily_for_horizon(machine)
    daily = build_fault_horizon_class_target(daily)

    feature_cols = [
        c for c in daily.columns
        if c not in {"DAY", "fault_flag_binary", "days_to_fault", "fault_horizon_class"}
        and pd.api.types.is_numeric_dtype(daily[c])
    ]

    usable = int(daily["fault_horizon_class"].notna().sum())
    if len(feature_cols) == 0 or usable < (seq_len + 10):
        print(f"[WARNING] Not enough valid data for horizon classification on {machine}.")
        return None

    Xh, yh, _ = make_horizon_tabular_sequences(daily, feature_cols, seq_len=seq_len)

    print_horizon_class_distribution(machine, yh)

    if yh.nunique() < 2:
        print(f"[WARNING] Horizon target for {machine} has fewer than 2 classes.")
        return None

    models = get_horizon_classifiers()
    pre, _, _ = build_preprocessor(Xh)

    results = []
    pipes = {}

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        try:
            r = evaluate_classifier(name, pipe, Xh, yh)
            results.append(r)
            pipes[name] = pipe
            print(f"[HORIZON] {machine} - {name}: F1={r['f1_macro']:.3f}, ACC={r['accuracy']:.3f}")
        except Exception as e:
            print(f"[WARNING] Skipping horizon model '{name}' for {machine}: {e}")

    if not results:
        return None

    leader = sorted(results, key=lambda d: (d["f1_macro"], d["accuracy"]), reverse=True)
    best_model = leader[0]["model"]
    pipes[best_model].fit(Xh, yh)

    bundle = {
        "machine": machine,
        "target": "fault_horizon_class",
        "horizon_definition": {
            "Risk": "0-10 days using 3 past daily inputs",
            "Safe": ">10 days using 3 past daily inputs",
        },
        "seq_len": seq_len,
        "feature_columns": list(Xh.columns),
        "class_labels": sorted(yh.astype(str).unique().tolist()),
        "best_model": best_model,
        "pipelines": pipes,
        "leaderboard": leader,
    }

    save_horizon_bundle(machine, bundle)
    return bundle


def main() -> None:
    results_dir = "results"
    ensure_dir(results_dir)

    summary_rows = []

    for m in MACHINES.keys():
        print(f"Training: {m}")
        bundle = train_machine(m)
        save_bundle(m, bundle)

        print(f"Saved bundle for: {m}")
        print("Best severity:", bundle["severity"]["best_model"])
        print("Best fault flag:", bundle["fault_flag"]["best_model"])
        print(
            "Best fault type:",
            bundle["fault_type"]["best_model"] if bundle["fault_type"]["best_model"] else "rule_based"
        )
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
                horizon_bundle = train_fault_horizon_classifier(
                    machine=m,
                    seq_len=3,
                )
                if horizon_bundle is not None:
                    save_horizon_results(horizon_bundle, m, results_dir)
                    print(f"[BEST HORIZON] {m}: {horizon_bundle['best_model']}")
                    print(f"[HORIZON CLASSES] {m}: {horizon_bundle['class_labels']}")
                    print(f"[HORIZON DEFINITION] {m}: {horizon_bundle['horizon_definition']}")
                    print("-" * 50)
            except Exception as e:
                print(f"[WARNING] Horizon classifier training failed for {m}: {e}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(results_dir, "overall_best_models_summary.csv"),
        index=False
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import os
import re
import warnings
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
    build_preprocessor,
    save_bundle,
    build_rul_target,
)

warnings.filterwarnings("ignore")

# ---------- Plot style ----------
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

    # Exact matches
    if name in {
        "DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME",
        "DAILY CONDITION", "DAILY CONDITION_1"
    }:
        return True

    # Prefix matches for variants like DAY_1, WEEK_1, DAILY CONDITION_2, etc.
    prefixes = (
        "DAY", "WEEK", "DATE", "TIME", "TIMESTAMP", "DATETIME",
        "DAILY CONDITION"
    )
    return any(name.startswith(prefix) for prefix in prefixes)


def should_exclude_feature(machine: str, col_name: str) -> bool:
    name = normalize_col_name(col_name)

    # Remove time-like / daily condition leakage columns
    if is_time_like_column(col_name):
        return True

    # Machine-specific exclusions
    if machine == "genset" and name == "REMARKS":
        return True

    if machine == "pellet" and name in {"DAILY CONDITION_1", "DAILY CONDITION"}:
        return True

    return False


def filter_feature_columns(machine: str, X: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in X.columns if should_exclude_feature(machine, c)]
    X = X.drop(columns=cols_to_drop, errors="ignore")
    X = X.dropna(axis=1, how="all")
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


def evaluate_classifier(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_macro = cross_val_score(
        pipe, X, y, cv=cv, scoring="f1_macro", error_score="raise"
    ).mean()

    acc = cross_val_score(
        pipe, X, y, cv=cv, scoring="accuracy", error_score="raise"
    ).mean()

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
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

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
    df = pd.DataFrame(report_dict).T
    df.to_csv(save_path, index=True)


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
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).head(top_n)

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
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
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
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


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

        csv_path = os.path.join(machine_dir, f"{machine}_{sec}_leaderboard.csv")
        df.to_csv(csv_path, index=False)

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

        csv_path = os.path.join(machine_dir, f"{machine}_rul_leaderboard.csv")
        df.to_csv(csv_path, index=False)

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


def train_machine(machine: str) -> Dict[str, Any]:
    cfg = MACHINES[machine]
    df = load_machine_df(machine)
    cond_col = cfg["condition_col"]
    time_key = cfg["time_key"]

    drop_cols = {cond_col, "condition_severity", "fault_type", "fault_flag"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Remove time-like leakage columns and machine-specific excluded inputs
    X = filter_feature_columns(machine, X)

    y_sev = df["condition_severity"].astype(str)
    y_fault_flag = df["fault_flag"].astype(str)

    fault_rows = df["fault_type"].astype(str) != "Normal"
    y_fault_type = df.loc[fault_rows, "fault_type"].astype(str)
    X_fault = X.loc[fault_rows].copy()

    pre, _, _ = build_preprocessor(X)
    clf_models = get_classifiers()

    # Severity
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

    # Fault flag
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

    # Fault type
    ft_leader = []
    ft_pipes = {}
    if len(X_fault) >= 30 and y_fault_type.nunique() >= 2:
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

    # RUL
    df_rul = build_rul_target(df, machine, time_key, cond_col)
    rul_bundle = None
    rul_leader = []

    if df_rul["rul_hours"].notna().sum() >= 60:
        Xr = df_rul.drop(
            columns=[cond_col, "condition_severity", "fault_type", "fault_flag", "rul_hours"],
            errors="ignore",
        )
        Xr = Xr.loc[df_rul["rul_hours"].notna()].copy()
        yr = df_rul.loc[df_rul["rul_hours"].notna(), "rul_hours"].values.astype(float)

        # Remove time-like leakage columns and machine-specific excluded inputs
        Xr = filter_feature_columns(machine, Xr)

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

    ft_best = None
    if ft_leader:
        ft_best = ft_leader[0]["model"]
        ft_pipes[ft_best].fit(X_fault, y_fault_type)

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
        },
        "rul": rul_bundle,
    }

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

        if bundle["fault_type"]["best_model"]:
            print("Best fault type:", bundle["fault_type"]["best_model"])

        if bundle["rul"]:
            print("Best RUL:", bundle["rul"]["best_model"])

        print("Final feature columns used:")
        print(bundle["feature_columns"])
        print("-" * 50)

        save_results(bundle, m, results_dir)

        summary_rows.append({
            "machine": m,
            "best_severity_model": bundle["severity"]["best_model"],
            "best_fault_flag_model": bundle["fault_flag"]["best_model"],
            "best_fault_type_model": bundle["fault_type"]["best_model"],
            "best_rul_model": bundle["rul"]["best_model"] if bundle["rul"] else None,
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(results_dir, "overall_best_models_summary.csv"),
        index=False
    )


if __name__ == "__main__":
    main()
\
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from joblib import dump
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_churn(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 72, size=n_samples)
    monthly_charges = rng.uniform(20, 120, size=n_samples).round(2)
    base_total = tenure * monthly_charges
    total_charges = (base_total + rng.normal(0, 150, size=n_samples)).clip(min=0).round(2)

    support_calls = rng.integers(0, 11, size=n_samples)
    data_usage_gb = rng.gamma(shape=2.0, scale=10.0, size=n_samples).round(2)

    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["credit_card", "bank_transfer", "electronic_check", "mailed_check"],
                                size=n_samples, p=[0.3, 0.25, 0.3, 0.15])
    internet_service = rng.choice(["dsl", "fiber", "none"], size=n_samples, p=[0.4, 0.5, 0.1])
    paperless_billing = rng.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    is_senior = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # True log-odds model (for simulation)
    # higher churn for: short tenure, high monthly_charges, many support_calls,
    # month-to-month, electronic_check, fiber, paperless_billing
    z = (
        -2.0
        + (-0.03) * tenure
        + (0.02) * monthly_charges
        + (0.18) * support_calls
        + (0.007) * (np.sqrt(np.maximum(data_usage_gb, 0)))
        + (0.5) * (contract_type == "month-to-month").astype(float)
        + (-0.4) * (contract_type == "two-year").astype(float)
        + (0.35) * (payment_method == "electronic_check").astype(float)
        + (0.25) * (internet_service == "fiber").astype(float)
        + (0.15) * paperless_billing
        + (0.1) * is_senior
    )
    p = 1 / (1 + np.exp(-z))
    churned = (rng.uniform(0, 1, size=n_samples) < p).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_calls": support_calls,
        "data_usage_gb": data_usage_gb,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "paperless_billing": paperless_billing,
        "is_senior": is_senior,
        "churned": churned,
    })
    return df


def load_data(input_csv: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(input_csv)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y


def build_pipeline(X: pd.DataFrame, class_weight: str = "balanced") -> Tuple[Pipeline, list, list]:
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    clf = LogisticRegression(
        solver="liblinear",    # supports L1/L2
        max_iter=1000,
        class_weight=class_weight
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("log", clf)
    ])

    param_grid = {
        "log__C": [0.01, 0.1, 1.0, 10.0],
        "log__penalty": ["l1", "l2"],
    }

    return pipe, param_grid, (numeric_cols + categorical_cols)


def evaluate_and_save(
    pipe: Pipeline,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    artifacts_dir: Path,
    optimize_threshold: str = "none"
) -> Dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Fit with CV to find best hyperparameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=param_grid_global, cv=cv, scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    best_pipe = gs.best_estimator_

    # Prediction probabilities
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_default = (proba_test >= 0.5).astype(int)

    # Default-threshold metrics
    metrics_default = {
        "accuracy": float(accuracy_score(y_test, y_pred_default)),
        "precision": float(precision_score(y_test, y_pred_default, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_default, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_default, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
    }

    # Confusion matrix (default)
    cm_def = confusion_matrix(y_test, y_pred_default)
    _plot_confusion(cm_def, ["0","1"], artifacts_dir / "confusion_matrix_default.png", title="Confusion Matrix (thr=0.5)")

    # ROC & PR curves
    _plot_roc(y_test, proba_test, artifacts_dir / "roc_curve.png")
    _plot_pr(y_test, proba_test, artifacts_dir / "pr_curve.png")

    # Threshold optimization (optional)
    opt = {"threshold": 0.5, "criterion": "none", "f1_at_opt": metrics_default["f1"]}
    y_pred_opt = y_pred_default
    if optimize_threshold in ("f1", "youden"):
        if optimize_threshold == "f1":
            prec, rec, thr = precision_recall_curve(y_test, proba_test)
            f1s = 2 * (prec * rec) / np.maximum(prec + rec, 1e-9)
            # precision_recall_curve returns thresholds of size n-1; align
            thr_candidates = np.append(thr, 1.0)
            f1s = f1s  # already aligned to prec/rec points
            best_idx = int(np.nanargmax(f1s))
            best_thr = float(thr_candidates[best_idx])
            opt["threshold"] = best_thr
            opt["criterion"] = "f1"
            y_pred_opt = (proba_test >= best_thr).astype(int)
            opt["f1_at_opt"] = float(f1_score(y_test, y_pred_opt, zero_division=0))

        elif optimize_threshold == "youden":
            fpr, tpr, thr = roc_curve(y_test, proba_test)
            J = tpr - fpr
            best_idx = int(np.nanargmax(J))
            best_thr = float(thr[best_idx])
            opt["threshold"] = best_thr
            opt["criterion"] = "youden_J"
            y_pred_opt = (proba_test >= best_thr).astype(int)
            opt["f1_at_opt"] = float(f1_score(y_test, y_pred_opt, zero_division=0))

        # Confusion (optimized)
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        _plot_confusion(cm_opt, ["0","1"], artifacts_dir / "confusion_matrix_optimized.png",
                        title=f"Confusion Matrix (thr={opt['threshold']:.3f}, {opt['criterion']})")

    # Classification report
    report = classification_report(y_test, y_pred_opt, zero_division=0)
    (artifacts_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Save model
    dump(best_pipe, artifacts_dir / "model.joblib")

    # Save coefficients
    _save_coef_importance(best_pipe, artifacts_dir / "coef_importance.csv")

    # Save metrics.json
    out = {
        "best_params": gs.best_params_,
        "opt_threshold": opt,
        "metrics_default_threshold": metrics_default,
        "metrics_at_opt_threshold": {
            "accuracy": float(accuracy_score(y_test, y_pred_opt)),
            "precision": float(precision_score(y_test, y_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_opt, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred_opt, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba_test)),
            "pr_auc": float(average_precision_score(y_test, proba_test)),
        },
        "n_test": int(len(y_test))
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _plot_confusion(cm: np.ndarray, classes, out_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_roc(y_true, proba, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_pr(y_true, proba, out_path: Path):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_coef_importance(best_pipe: Pipeline, out_csv: Path):
    # Extract feature names after preprocessing
    preprocess = best_pipe.named_steps["preprocess"]
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        # Fallback: unnamed
        feature_names = [f"f{i}" for i in range(best_pipe.named_steps["log"].coef_.shape[1])]

    coef = best_pipe.named_steps["log"].coef_[0]
    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)
    df_coef.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Practical Logistic Regression (train/eval/artifacts)")
    parser.add_argument("--input_csv", type=str, default="", help="입력 CSV 경로(미지정시 시뮬레이션 데이터 사용)")
    parser.add_argument("--target", type=str, default="churned", help="타깃 컬럼명")
    parser.add_argument("--test_size", type=float, default=0.2, help="테스트 비율")
    parser.add_argument("--random_state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="결과 저장 폴더")
    parser.add_argument("--optimize_threshold", type=str, default="none", choices=["none","f1","youden"],
                        help="임계값 최적화 기준")
    parser.add_argument("--class_weight", type=str, default="balanced", help="로지스틱 회귀 class_weight")
    args = parser.parse_args()

    rng = np.random.RandomState(args.random_state)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load or simulate data
    if args.input_csv:
        X, y = load_data(Path(args.input_csv), args.target)
    else:
        df = simulate_churn(n_samples=5000, random_state=args.random_state)
        X, y = df.drop(columns=[args.target]), df[args.target]

    # Build pipe & grid
    global param_grid_global
    pipe, param_grid_global, cols = build_pipeline(X, class_weight=args.class_weight)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train/Eval/Save
    results = evaluate_and_save(pipe, X_train, y_train, X_test, y_test, artifacts_dir, args.optimize_threshold)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

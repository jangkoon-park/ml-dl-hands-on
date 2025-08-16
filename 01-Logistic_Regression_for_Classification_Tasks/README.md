# **English Version**

# 01 — Logistic Regression for Classification (Practical Example)

**Goal**: Assume a general churn prediction problem, and provide a fully reproducible practical pipeline from **preprocessing → training/tuning → evaluation → artifact saving**.

## Folder Structure

```
01-Logistic_Regression_for_Classification_Tasks/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sample_churn.csv                # Sample dataset (synthetic customer churn data)
├─ artifacts/                         # Outputs generated after running
│  ├─ model.joblib
│  ├─ metrics.json
│  ├─ classification_report.txt
│  ├─ coef_importance.csv
│  ├─ confusion_matrix_default.png
│  ├─ confusion_matrix_optimized.png
│  ├─ roc_curve.png
│  └─ pr_curve.png
└─ src/
   ├─ train_eval.py                   # Training/Evaluation/Artifact saving (practical)
   └─ plot_boundary.py                # 2D demo (decision boundary visualization)
```

## 1) Quick Run (Reproduce with Sample Data)

```bash
# (Optional) Virtual environment
# python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

# Train/Evaluate/Save with sample data
python src/train_eval.py   --input_csv data/sample_churn.csv   --target churned   --artifacts_dir artifacts   --optimize_threshold f1   --random_state 42
```

* **Artifacts** will be saved under `artifacts/`.
* Key metrics (Accuracy, ROC-AUC, PR-AUC, F1, etc.) are written to `metrics.json`.
* Feature importance coefficients are exported to `coef_importance.csv`.

## 2) Apply to Real Data

CSV schema example (flexible column names, but mixed numeric + categorical is expected):

* Numeric: `tenure`, `monthly_charges`, `total_charges`, `support_calls`, `data_usage_gb`
* Categorical: `contract_type`, `payment_method`, `internet_service`, `paperless_billing`, `is_senior`
* Target: `churned` (0/1)

Example:

```bash
python src/train_eval.py   --input_csv /path/to/your/data.csv   --target churned   --artifacts_dir ./artifacts_real   --optimize_threshold youden   --random_state 2025
```

## 3) 2D Decision Boundary Demo (for Presentation)

```bash
python src/plot_boundary.py --samples 600 --C 1.0 --noise 0.2 --out artifacts/decision_boundary.png
```

## 4) Design Points (Practical Checklist)

* **ColumnTransformer**: Separate numeric/categorical pipelines (Impute/Scale/OneHot)
* **GridSearchCV**: Tune `C` and `penalty (L1/L2)` with Stratified K-Fold
* **Class Weight** (optional) to handle imbalance
* Save **ROC/PR Curve**, **Confusion Matrices** (default/optimized threshold)
* Save **coef\_importance.csv** for feature influence interpretation
* Save model with **joblib** for deployment/reuse

---

# **한국어 Version**

# 01 — 로지스틱 회귀 분류 (실무형 예제)

**목표**: 범용적인 고객 이탈(Churn) 예측 문제를 가정하고, **전처리 → 학습/튜닝 → 평가 → 아티팩트 저장**까지 한 번에 재현 가능한 실무형 파이프라인을 제공합니다.

## 폴더 구조

```
01-Logistic_Regression_for_Classification_Tasks/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sample_churn.csv                # 샘플 데이터(가상의 고객 이탈 데이터)
├─ artifacts/                         # 실행 후 생성되는 결과물 폴더
│  ├─ model.joblib
│  ├─ metrics.json
│  ├─ classification_report.txt
│  ├─ coef_importance.csv
│  ├─ confusion_matrix_default.png
│  ├─ confusion_matrix_optimized.png
│  ├─ roc_curve.png
│  └─ pr_curve.png
└─ src/
   ├─ train_eval.py                   # 학습/평가/아티팩트 저장 (실무형)
   └─ plot_boundary.py                # 2D 데모(의사결정 경계 시각화)
```

## 1) 빠른 실행 (샘플 데이터로 재현)

```bash
# (선택) 가상환경
# python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

# 샘플 데이터로 학습/평가/저장
python src/train_eval.py   --input_csv data/sample_churn.csv   --target churned   --artifacts_dir artifacts   --optimize_threshold f1   --random_state 42
```

* **아티팩트**는 `artifacts/`에 저장됩니다.
* 주요 지표(Accuracy, ROC-AUC, PR-AUC, F1 등)는 `metrics.json`에 기록됩니다.
* 계수 중요도는 `coef_importance.csv`로 저장되어 특성 영향 해석이 가능합니다.

## 2) 실데이터 적용

CSV 스키마 예시 (컬럼명은 자유, 단 **수치 + 범주형 혼합** 구조 필요):

* 수치형: `tenure`, `monthly_charges`, `total_charges`, `support_calls`, `data_usage_gb`
* 범주형: `contract_type`, `payment_method`, `internet_service`, `paperless_billing`, `is_senior`
* 타깃: `churned` (0/1)

예시:

```bash
python src/train_eval.py   --input_csv /path/to/your/data.csv   --target churned   --artifacts_dir ./artifacts_real   --optimize_threshold youden   --random_state 2025
```

## 3) 2D 결정 경계 데모(설명/발표용)

```bash
python src/plot_boundary.py --samples 600 --C 1.0 --noise 0.2 --out artifacts/decision_boundary.png
```

## 4) 설계 포인트 (실무 체크리스트)

* **ColumnTransformer**로 수치/범주형 파이프라인 분리 (Impute/Scale/OneHot)
* **GridSearchCV**로 `C`, `penalty (L1/L2)` 튜닝 (계층 K-Fold)
* **Class Weight** 옵션으로 불균형 데이터 대응
* **ROC/PR Curve**, **혼동행렬(Confusion Matrix)** 저장 (기본/최적 임계값)
* **coef\_importance.csv** 저장 → 특성 영향 해석
* 모델을 **joblib**으로 저장 → 배포/재사용 가능

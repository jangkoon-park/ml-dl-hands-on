
````md
01-Logistic_Regression_for_Classification_Tasks/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ train_eval.py          # training + evaluation + artifact saving
│  └─ plot_boundary.py       # decision boundary visualization (demo)
└─ results/                  # generated after execution (model/metrics/plots)


# 01 — Logistic Regression for Classification Tasks

**Goal:** Binary classification (Breast Cancer) with Logistic Regression  
**Deliverables:** `results/metrics.json`, `results/roc_curve.png`, `results/confusion_matrix.png`, `results/model.joblib`, `results/coef_importance.csv`

## Quickstart (Windows, CMD)

```bat
cd 01-Logistic_Regression_for_Classification_Tasks
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src\train_eval.py --C 1.0 --penalty l2 --class_weight none --test_size 0.2 --random_state 42
python src\plot_boundary.py --feature_x "mean radius" --feature_y "mean texture"
````

---

# 01 — 로지스틱 회귀 분류 과제

**목표:** 유방암 데이터셋을 활용한 이진 분류 (Logistic Regression)
**산출물:** `results/metrics.json`, `results/roc_curve.png`, `results/confusion_matrix.png`, `results/model.joblib`, `results/coef_importance.csv`

## 빠른 실행 (Windows, CMD)

```bat
cd 01-Logistic_Regression_for_Classification_Tasks
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src\train_eval.py --C 1.0 --penalty l2 --class_weight none --test_size 0.2 --random_state 42
python src\plot_boundary.py --feature_x "mean radius" --feature_y "mean texture"
```

```


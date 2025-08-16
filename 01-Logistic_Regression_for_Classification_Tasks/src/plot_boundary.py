\
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def plot_decision_boundary(model, X, y, out_path):
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)[:,1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5,4))
    cs = ax.contourf(xx, yy, Z, alpha=0.6, levels=20)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("P(y=1)")
    ax.scatter(X[:,0], X[:,1], c=y, s=20, edgecolors="k")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Logistic Regression Decision Boundary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="2D Decision Boundary Demo for Logistic Regression")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="decision_boundary.png")
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.samples, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.0, flip_y=args.noise, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("log", LogisticRegression(C=args.C, solver="lbfgs"))
    ])
    pipe.fit(X, y)

    plot_decision_boundary(pipe, X, y, args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

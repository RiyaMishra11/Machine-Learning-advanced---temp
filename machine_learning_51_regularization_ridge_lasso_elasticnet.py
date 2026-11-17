# ML 51: Ridge, Lasso and ElasticNet
# Install: pip install scikit-learn pandas
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    X, y = make_regression(n_samples=500, n_features=12, n_informative=7,
                           noise=18, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {"Ridge": Ridge(alpha=1), "Lasso": Lasso(alpha=0.1, max_iter=10000),
              "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)}
    rows = []
    for name, estimator in models.items():
        model = make_pipeline(StandardScaler(), estimator).fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"model": name, "MSE": mean_squared_error(yte, pred),
                     "R2": r2_score(yte, pred)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()

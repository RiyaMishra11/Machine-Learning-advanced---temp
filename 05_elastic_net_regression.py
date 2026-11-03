"""225 — Elastic Net Regression"""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def main():
    d = load_diabetes()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, random_state=42)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000, random_state=42))
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    print("Elastic Net Regression")
    print(f"MAE: {mean_absolute_error(yte,p):.2f}")
    print(f"RMSE: {mean_squared_error(yte,p)**0.5:.2f}")
    print(f"R²: {r2_score(yte,p):.3f}")
    print("Non-zero coefficients:", int(np.count_nonzero(model.named_steps["elasticnet"].coef_)))

if __name__ == "__main__":
    main()

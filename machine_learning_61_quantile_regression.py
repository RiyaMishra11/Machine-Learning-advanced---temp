"""ML 61: Quantile Regression."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_pinball_loss

def main():
    X, y = make_regression(n_samples=800, n_features=7, noise=25, random_state=61)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=61)
    preds = {}
    for q in (0.10, 0.50, 0.90):
        model = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=120,
                                          learning_rate=0.05, max_depth=2, random_state=61)
        model.fit(Xtr, ytr)
        preds[q] = model.predict(Xte)
        print(f"Quantile {q:.0%} pinball loss: {mean_pinball_loss(yte, preds[q], alpha=q):.3f}")
    coverage = np.mean((yte >= preds[0.10]) & (yte <= preds[0.90]))
    print(f"Observed 10%-90% interval coverage: {coverage:.1%}")
    print("Median predictions:", np.round(preds[0.50][:5], 2))

if __name__ == "__main__":
    main()

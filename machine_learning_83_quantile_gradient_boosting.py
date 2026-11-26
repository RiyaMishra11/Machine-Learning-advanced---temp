"""ML 83: Quantile Gradient Boosting for prediction intervals."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

def main():
    X, y = make_regression(n_samples=900, n_features=10, noise=35, random_state=83)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=83)
    predictions = {}
    for q in (0.1, 0.5, 0.9):
        model = GradientBoostingRegressor(loss="quantile", alpha=q,
                                           n_estimators=160, max_depth=2,
                                           learning_rate=0.05, random_state=83)
        model.fit(X_train, y_train)
        predictions[q] = model.predict(X_test)
        print(f"Quantile {q:.1f} pinball loss:",
              round(mean_pinball_loss(y_test, predictions[q], alpha=q), 2))
    coverage = np.mean((y_test >= predictions[0.1]) & (y_test <= predictions[0.9]))
    print(f"Observed 10%-90% interval coverage: {coverage:.3f}")
    print("Median predictions:", np.round(predictions[0.5][:5], 2))

if __name__ == "__main__":
    main()

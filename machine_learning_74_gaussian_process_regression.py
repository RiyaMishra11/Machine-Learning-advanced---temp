"""ML 74: Gaussian Process Regression with uncertainty."""
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_friedman1(n_samples=350, n_features=4, noise=1.5, random_state=74)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=74)
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.5)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                     n_restarts_optimizer=1, random_state=74)
    model.fit(Xtr, ytr)
    pred, std = model.predict(Xte, return_std=True)
    print("Gaussian Process Regression")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")
    print(f"Mean predictive standard deviation: {std.mean():.3f}")
    for actual, estimate, uncertainty in zip(yte[:5], pred[:5], std[:5]):
        print(f"Actual={actual:.2f}, predicted={estimate:.2f} +/- {1.96*uncertainty:.2f}")

if __name__ == "__main__":
    main()

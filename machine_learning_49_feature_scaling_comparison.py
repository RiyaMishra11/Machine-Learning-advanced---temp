# ML 49: Feature Scaling Comparison
# Install: pip install scikit-learn pandas
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def main():
    X, _ = make_classification(n_samples=300, n_features=5, random_state=42)
    X[:5, 0] *= 12  # Add a few extreme values to demonstrate outlier sensitivity.
    scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(),
               "Robust": RobustScaler()}
    rows = []
    for name, scaler in scalers.items():
        z = scaler.fit_transform(X)[:, 0]
        rows.append({"scaler": name, "mean": z.mean(), "std": z.std(),
                     "min": z.min(), "max": z.max()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()

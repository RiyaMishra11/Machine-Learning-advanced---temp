"""
Machine Learning File 85: Feature Scaling
Demonstrates StandardScaler and MinMaxScaler using the Wine dataset.
Install: pip install scikit-learn pandas
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    wine = load_wine(as_frame=True)
    X = wine.data

    standard_scaled = StandardScaler().fit_transform(X)
    minmax_scaled = MinMaxScaler().fit_transform(X)

    standard_df = pd.DataFrame(standard_scaled, columns=X.columns)
    minmax_df = pd.DataFrame(minmax_scaled, columns=X.columns)

    print("Original data (first 5 rows):")
    print(X.head())
    print("\nStandardScaler output (first 5 rows):")
    print(standard_df.head().round(3))
    print("\nMinMaxScaler output (first 5 rows):")
    print(minmax_df.head().round(3))

    print("\nStandardized column means (approximately 0):")
    print(standard_df.mean().round(3))
    print("\nMinMax scaled range:")
    print(f"Minimum: {minmax_df.min().min():.3f}")
    print(f"Maximum: {minmax_df.max().max():.3f}")


if __name__ == "__main__":
    main()

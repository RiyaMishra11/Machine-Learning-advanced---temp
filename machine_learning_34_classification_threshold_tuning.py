"""ML 34: Compare precision and recall at different thresholds."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    X, y = make_classification(n_samples=1200, n_features=12, n_informative=7,
                               n_redundant=2, weights=[0.75, 0.25], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    print("Threshold | Precision | Recall | F1")
    for threshold in np.arange(0.2, 0.81, 0.1):
        pred = (probabilities >= threshold).astype(int)
        print(f"{threshold:8.2f} | {precision_score(y_test, pred, zero_division=0):9.3f} | "
              f"{recall_score(y_test, pred, zero_division=0):6.3f} | "
              f"{f1_score(y_test, pred, zero_division=0):.3f}")

if __name__ == "__main__":
    main()

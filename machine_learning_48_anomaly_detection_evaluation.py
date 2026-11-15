"""ML 48: Evaluate anomaly detection. pip install scikit-learn numpy"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

def main():
    normal, _ = make_blobs(n_samples=500, centers=2, cluster_std=0.75, random_state=42)
    unusual, _ = make_blobs(n_samples=35, centers=1, cluster_std=2.8,
                            center_box=(8.0, 12.0), random_state=7)
    X = np.vstack([normal, unusual])
    y_true = np.array([0] * len(normal) + [1] * len(unusual))
    X = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=0.07, random_state=42).fit(X)
    y_pred = (model.predict(X) == -1).astype(int)
    scores = -model.decision_function(X)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"],
                                zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_true, scores), 3))

if __name__ == "__main__":
    main()

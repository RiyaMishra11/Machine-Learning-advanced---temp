"""ML 28: Learning Curve for Classification"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
import numpy as np

def main():
    data = load_breast_cancer()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, train_scores, val_scores = learning_curve(
        model, data.data, data.target,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv, scoring="accuracy", n_jobs=-1)
    print("Training size | Train accuracy | Validation accuracy")
    for size, train_row, val_row in zip(sizes, train_scores, val_scores):
        print(f"{size:13d} | {train_row.mean():14.4f} | {val_row.mean():19.4f}")

if __name__ == "__main__":
    main()

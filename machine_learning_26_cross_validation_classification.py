"""ML 26: Stratified K-Fold Cross-Validation"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def main():
    data = load_breast_cancer()
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, data.data, data.target, cv=cv, scoring="accuracy")
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: {score:.4f}")
    print(f"Mean accuracy: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")

if __name__ == "__main__":
    main()

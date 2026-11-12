"""ML 32: End-to-end classification pipeline."""
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2,
                                           random_state=42, stratify=data.target)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, Xtr, ytr, cv=cv, scoring="accuracy")
    pipeline.fit(Xtr, ytr)
    pred = pipeline.predict(Xte)
    print("CV scores:", [round(s, 4) for s in scores])
    print("Mean CV accuracy:", round(scores.mean(), 4))
    print("Test accuracy:", round(accuracy_score(yte, pred), 4))
    print(classification_report(yte, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

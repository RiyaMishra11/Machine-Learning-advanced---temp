"""ML 24: Compare default and class-weighted models on imbalanced data."""
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score

def main():
    X, y = make_classification(
        n_samples=3000, n_features=15, n_informative=6, n_redundant=3,
        n_classes=2, weights=[0.92, 0.08], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "Default Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),
        "Class-Weighted Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"\n{name}")
        print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 4))
        print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()

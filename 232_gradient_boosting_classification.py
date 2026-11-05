"""232: Gradient Boosting Classification."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("GRADIENT BOOSTING CLASSIFICATION")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=data.target_names))
    print("\nTop feature importances:")
    for name, score in sorted(zip(data.feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name:30s} {score:.4f}")

if __name__ == "__main__":
    main()

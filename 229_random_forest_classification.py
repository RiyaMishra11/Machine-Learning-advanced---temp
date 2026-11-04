"""File 229: Random Forest Classification with feature importance."""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("RANDOM FOREST CLASSIFICATION")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Classification report:\n",
          classification_report(y_test, pred, target_names=data.target_names))
    print("\nTop feature importances:")
    ranked = sorted(zip(data.feature_names, model.feature_importances_),
                    key=lambda item: item[1], reverse=True)
    for name, score in ranked[:10]:
        print(f"{name:25s} {score:.4f}")

if __name__ == "__main__":
    main()

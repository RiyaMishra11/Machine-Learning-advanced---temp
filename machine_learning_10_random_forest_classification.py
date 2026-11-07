"""Machine Learning 10: Random Forest Classification."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("MACHINE LEARNING 10 - RANDOM FOREST")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=data.target_names))
    print("Top 10 feature importances:")
    for name, score in sorted(zip(data.feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main()

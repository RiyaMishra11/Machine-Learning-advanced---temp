"""233: AdaBoost Classification."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=42, stratify=data.target)
    weak_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(estimator=weak_tree, n_estimators=100, learning_rate=0.5, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("ADABOOST CLASSIFICATION")
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

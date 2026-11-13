"""ML 35: Multiclass classification using One-vs-Rest and One-vs-One."""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\\n{name} accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, zero_division=0))

def main():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )
    base = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    evaluate("One-vs-Rest", OneVsRestClassifier(base), X_train, X_test, y_train, y_test)
    evaluate("One-vs-One", OneVsOneClassifier(base), X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

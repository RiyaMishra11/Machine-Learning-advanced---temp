"""
Machine Learning 16: Stacking Classifier
Use multiple base models and a Logistic Regression meta-model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    base_models = [
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))),
        ("forest", RandomForestClassifier(n_estimators=150, random_state=42)),
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Stacking Classifier")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:\n", classification_report(
        y_test, predictions, target_names=data.target_names
    ))


if __name__ == "__main__":
    main()

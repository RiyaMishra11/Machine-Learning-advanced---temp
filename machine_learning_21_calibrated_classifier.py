"""ML 21: Probability calibration for an SVM classifier."""
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    base = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    print("Calibrated Classifier")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Log loss:", round(log_loss(y_test, proba), 4))
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

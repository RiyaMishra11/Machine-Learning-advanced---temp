"""File 226: Logistic Regression Classification on the Wine dataset."""
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, random_state=42))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("LOGISTIC REGRESSION CLASSIFICATION")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Classification report:\n",
          classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

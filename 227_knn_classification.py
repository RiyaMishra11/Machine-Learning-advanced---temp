"""File 227: K-Nearest Neighbors Classification on the Wine dataset."""
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    model = make_pipeline(StandardScaler(),
                          KNeighborsClassifier(n_neighbors=5, weights="distance"))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("K-NEAREST NEIGHBORS CLASSIFICATION")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Classification report:\n",
          classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

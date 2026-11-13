"""ML 33: Save and reload a trained model with joblib."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=42
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    path = Path("breast_cancer_model.joblib")
    joblib.dump(model, path)
    loaded = joblib.load(path)
    print("Saved model:", path.resolve())
    print("Reloaded model accuracy:", round(accuracy_score(y_test, loaded.predict(X_test)), 3))

if __name__ == "__main__":
    main()

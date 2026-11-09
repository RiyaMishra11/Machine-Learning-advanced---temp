"""ML 20: Quadratic Discriminant Analysis (QDA)"""
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    model = make_pipeline(StandardScaler(),
                          QuadraticDiscriminantAnalysis(reg_param=0.1))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Quadratic Discriminant Analysis")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

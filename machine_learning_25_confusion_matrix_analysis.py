"""ML 25: Confusion Matrix Analysis"""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = matrix.ravel()
    print("Confusion matrix:\n", matrix)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall   :", round(recall_score(y_test, pred), 4))
    print("F1 score :", round(f1_score(y_test, pred), 4))

if __name__ == "__main__":
    main()

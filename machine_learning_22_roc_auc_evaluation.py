"""ML 22: Evaluate a classifier using ROC-AUC and related metrics."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred), 4))
    print("Recall   :", round(recall_score(y_test, pred), 4))
    print("F1 score :", round(f1_score(y_test, pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, proba), 4))
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    print("\nFirst ROC points (FPR, TPR, threshold):")
    for row in zip(fpr[:5], tpr[:5], thresholds[:5]):
        print(tuple(round(float(v), 4) for v in row))

if __name__ == "__main__":
    main()

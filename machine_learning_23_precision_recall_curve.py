"""ML 23: Precision-recall curve and average precision."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    print("Average precision:", round(average_precision_score(y_test, scores), 4))
    print("\nSample threshold / precision / recall:")
    for threshold, p, r in zip(thresholds[:10], precision[:10], recall[:10]):
        print(f"{threshold:.3f} / {p:.3f} / {r:.3f}")

if __name__ == "__main__":
    main()

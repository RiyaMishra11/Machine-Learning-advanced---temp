"""ML 36: Multi-label classification with One-vs-Rest logistic regression."""
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    X, y = make_multilabel_classification(
        n_samples=1000, n_features=20, n_classes=4, n_labels=2,
        allow_unlabeled=False, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = OneVsRestClassifier(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Hamming loss:", round(hamming_loss(y_test, pred), 3))
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()

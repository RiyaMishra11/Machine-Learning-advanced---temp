"""ML 30: Recursive Feature Elimination (RFE)."""
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2,
                                           random_state=42, stratify=data.target)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    selector = RFE(estimator, n_features_to_select=10)
    Xtr_sel = selector.fit_transform(Xtr, ytr)
    Xte_sel = selector.transform(Xte)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(Xtr_sel, ytr)
    print("Selected features:", list(data.feature_names[selector.support_]))
    print("Accuracy:", round(accuracy_score(yte, model.predict(Xte_sel)), 4))

if __name__ == "__main__":
    main()

"""ML 31: Principal Component Analysis with Logistic Regression."""
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2,
                                           random_state=42, stratify=data.target)
    model = make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          LogisticRegression(max_iter=3000))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Original features:", Xtr.shape[1])
    print("PCA components:", model.named_steps["pca"].n_components_)
    print("Accuracy:", round(accuracy_score(yte, pred), 4))
    print(classification_report(yte, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()

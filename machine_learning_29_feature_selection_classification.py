"""ML 29: Feature selection using SelectKBest."""
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    data = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2,
                                           random_state=42, stratify=data.target)
    selector = SelectKBest(mutual_info_classif, k=10)
    Xtr = selector.fit_transform(Xtr, ytr)
    Xte = selector.transform(Xte)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(Xtr, ytr)
    print("Selected features:")
    print(list(data.feature_names[selector.get_support()]))
    print("Accuracy:", round(accuracy_score(yte, model.predict(Xte)), 4))

if __name__ == "__main__":
    main()

"""
File 090: Feature Selection
Topic: Select useful input features with a statistical scoring method.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load a binary classification dataset
X, y = load_breast_cancer(return_X_y=True)

# Split before feature selection to avoid using test labels during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Select the 10 features with the strongest ANOVA F-scores
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train a model using only the selected features
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
model.fit(X_train_selected, y_train)

predictions = model.predict(X_test_selected)
print("Original number of features:", X.shape[1])
print("Selected number of features:", X_train_selected.shape[1])
print("Selected feature indexes:", selector.get_support(indices=True))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))

"""
File 091: Hyperparameter Tuning with GridSearchCV
Topic: Find a strong model configuration by testing a parameter grid.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)

# GridSearchCV evaluates each combination using cross-validation
parameter_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
predictions = best_model.predict(X_test)

print("Best parameters:", search.best_params_)
print("Best cross-validation score:", round(search.best_score_, 4))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:\n", classification_report(y_test, predictions))

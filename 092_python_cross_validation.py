"""
File 092: Cross-Validation
Topic: Estimate model performance across multiple train/validation splits.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

X, y = load_iris(return_X_y=True)

# StratifiedKFold preserves class proportions in each fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# A pipeline keeps scaling inside each training fold and avoids data leakage
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv_strategy,
    scoring="accuracy",
)

print("Accuracy in each fold:", np.round(scores, 4))
print("Mean cross-validation accuracy:", round(scores.mean(), 4))
print("Standard deviation:", round(scores.std(), 4))

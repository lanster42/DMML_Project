import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pipeline ensures that scaling is done correctly inside CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Support Vector Machine classifier
from sklearn.svm import SVC

# Model selection utilities
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

# Evaluation metrics
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve
)


# 1. Load and prepare data

# Load training data (no header in CSV)
df = pd.read_csv("trainData/42_train.csv", header=None)

# Create binary labels:
# first 2000 samples -> class 0
# remaining samples  -> class 1
df["label"] = 0
df.loc[2000:, "label"] = 1

# Load test data (no labels)
test_data = pd.read_csv("trainData/42_test.csv", header=None)

# Split features and target
X = df.iloc[:, :-1]   # all columns except label
y = df.iloc[:, -1]    # label column


# 2. Baseline cross-validated AUC (no tuning)

# Pipeline:
# 1) StandardScaler is REQUIRED for SVMs (feature scale matters)
# 2) SVC with RBF kernel (default strong choice)
pipe_base = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,        # needed for ROC/AUC and submission scores
        class_weight="balanced", # handles class imbalance automatically
        random_state=42
    ))
])

# Stratified CV preserves class ratio in each fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)

# Cross-validated ROC AUC estimate
cv_auc = cross_val_score(
    pipe_base,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("5-fold CV AUCs:", np.round(cv_auc, 4))
print("Estimated AUC (mean ± std):", cv_auc.mean(), "±", cv_auc.std())


# 3. Train / validation split

# Hold-out validation set for final evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,   # preserve class balance
    random_state=0
)


# 4. Hyperparameter tuning (C and gamma only)

# For RBF SVMs, ONLY C and gamma matter
param_grid = {
    "svc__C": [0.1, 1, 10],            # regularization strength
    "svc__gamma": ["scale", 0.01, 0.1] # kernel width
}

# Grid search with ROC AUC as objective
gs = GridSearchCV(
    estimator=pipe_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# Fit grid search on training data only
gs.fit(X_train, y_train)

print("Best parameters:", gs.best_params_)
print("Best CV AUC:", gs.best_score_)

# Best trained pipeline
clf = gs.best_estimator_


# 5. Validation evaluation

# Hard predictions (class labels)
y_val_pred = clf.predict(X_val)

print(classification_report(y_val, y_val_pred))

# Soft predictions (probabilities for ROC/AUC)
y_val_scores = clf.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, y_val_scores)
print("Validation AUC:", val_auc)

# ROC curve
fpr, tpr, _ = roc_curve(y_val, y_val_scores)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {val_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")  # random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.show()


# 6. Final model training (full dataset)

# Retrain best model on ALL available labeled data
clf.fit(X, y)

# Predict probabilities for test data
test_scores = clf.predict_proba(test_data)[:, 1]

# Save submission file
pd.DataFrame(test_scores).to_csv(
    "42_submission_svm.csv",
    index=False,
    header=False
)

print("Submission shape should be (50000,):", test_scores.shape)


# 7. Overfitting check
# Compare train vs validation AUC
train_auc = roc_auc_score(
    y_train,
    clf.predict_proba(X_train)[:, 1]
)

print("Train AUC:", train_auc)
print("Val AUC: ", val_auc)
print("Gap:", train_auc - val_auc)


# 8. Label permutation test (sanity check)
# Randomly shuffle labels (should give AUC ≈ 0.5)
y_perm = y.sample(frac=1.0, random_state=42).values

perm_auc = cross_val_score(
    pipe_base,
    X,
    y_perm,
    cv=cv,
    scoring="roc_auc"
)

print("Permutation test AUCs:", perm_auc)
print("Permutation test mean AUC:", perm_auc.mean())

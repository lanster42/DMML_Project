import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay



#Reading the file with train data and adding a label column
df = pd.read_csv('42_train.csv', header=None)
df['label'] = 0
df.loc[2000:, 'label'] = 1

#Reading the file with test data
test_data = pd.read_csv('42_test.csv', header=None)

print(df.head())
print(df.tail())

#Splitting dataset in features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X.shape)
print(y.shape)

#Cross-validated AUC estimate
base_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0)
#5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_auc = cross_val_score(base_clf, X, y, cv=cv, scoring="roc_auc")
print("5-fold CV AUCs:", np.round(cv_auc, 4))
print("Estimated AUC (mean ± std):", cv_auc.mean(), "±", cv_auc.std())


#Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

#Rewighting the training samples
w0 = 0.4/0.5
w1 = 0.6/0.5
w = np.where(y_train == 0, w0, w1)

#Building decision tree model
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=0)
clf.fit(X_train, y_train, sample_weight=w)

#predicting the labels for the test data
y_val_pred = clf.predict(X_val)
print(metrics.classification_report(y_val, y_val_pred))
print((y_val_pred[1:6]))

#Visualizing the decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    filled=True,
    rounded=True,
    feature_names=[f"x{i}" for i in range(X.shape[1])],
    class_names=['0', '1'],
    fontsize=8
)
plt.title("Decision Tree")
plt.show()

#ROC + AUC on validation
y_val_scores = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_scores)
print("Validation AUC:", auc)

#ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.show()


#Final model
w_full = np.where(y==0, w0, w1)
clf.fit(X, y, sample_weight=w_full)

X_test = test_data
test_scores = clf.predict_proba(X_test)[:, 1]

#Submition file
pd.DataFrame(test_scores).to_csv("42_submission.csv", index=False, header=False)
print("Submission shape should be (50000,):", test_scores.shape)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay



#Reading the file with train data and adding a label column
df = pd.read_csv('trainData/42_train.csv', header=None)
df['label'] = 0
df.loc[2000:, 'label'] = 1

#Reading the file with test data
test_data = pd.read_csv('trainData/42_test.csv', header=None)


#Splitting training set in features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#build pipeline: scaler -> logistic regression (for data scaling)
pipe = Pipeline([       #the fact that scaling the data didn't change the result means that data is likely already scaled well
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=0, max_iter=2000))
])

#cross-validation AUC with pipeline
kl = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
kl_auc = cross_val_score(pipe, X, y, cv=kl, scoring='roc_auc')
print("5-fold CV AUCs:", np.round(kl_auc, 4))
print("Estimated AUC (mean ± std):", kl_auc.mean(), "±", kl_auc.std())


#Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

#Rewighting the training samples
w0 = 0.4/0.5
w1 = 0.6/0.5
w = np.where(y_train == 0, w0, w1)

#Building logistic regression model
pipe.fit(X_train, y_train, logreg__sample_weight=w)

#predicting the labels for the test data
y_val_pred = pipe.predict(X_val)
print(metrics.classification_report(y_val, y_val_pred))
print((y_val_pred[1:6]))


#ROC + AUC on validation
y_val_scores = pipe.predict_proba(X_val)[:, 1]
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
pipe.fit(X, y, logreg__sample_weight=w_full)

X_test = test_data
test_scores = pipe.predict_proba(X_test)[:, 1]

#Submition file
pd.DataFrame(test_scores).to_csv("42_submission_logist_reg.csv", index=False, header=False)
print("Submission shape should be (50000,):", test_scores.shape)
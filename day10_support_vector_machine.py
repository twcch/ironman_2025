import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)
from sklearn.svm import SVC, LinearSVC

# 1) 載入資料（10 類、多分類）
X, y = load_digits(return_X_y=True)

# 2) 訓練／測試分割（分層）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) 基線：LinearSVC（速度快）——記得標準化；提高 max_iter 避免收斂警告
linear_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(C=1.0, random_state=42, max_iter=5000))
])

linear_clf.fit(X_train, y_train)
y_pred_linear = linear_clf.predict(X_test)

linear_acc = accuracy_score(y_test, y_pred_linear)
linear_macro_f1 = f1_score(y_test, y_pred_linear, average="macro")
linear_weighted_f1 = f1_score(y_test, y_pred_linear, average="weighted")

print("=== LinearSVC (baseline) ===")
print("Accuracy:", round(linear_acc, 4))
print("Macro-F1:", round(linear_macro_f1, 4))
print("Weighted-F1:", round(linear_weighted_f1, 4))
print(classification_report(y_test, y_pred_linear, digits=4))

# 4) RBF SVC（非線性）——小網格搜尋；若你覺得慢，可把 probability=False
rbf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42))  # 如需更快：probability=False
])

param_grid = {
    "clf__C": [1, 10, 100],
    "clf__gamma": ["scale", 0.01]
}
rbf_gs = GridSearchCV(
    rbf_pipe, param_grid=param_grid, cv=3,
    scoring="f1_weighted", n_jobs=-1, verbose=0
)
rbf_gs.fit(X_train, y_train)

best_rbf = rbf_gs.best_estimator_
y_pred_rbf = best_rbf.predict(X_test)

rbf_acc = accuracy_score(y_test, y_pred_rbf)
rbf_macro_f1 = f1_score(y_test, y_pred_rbf, average="macro")
rbf_weighted_f1 = f1_score(y_test, y_pred_rbf, average="weighted")

print("\n=== RBF SVC (tuned) ===")
print("Best params:", rbf_gs.best_params_)
print("Accuracy:", round(rbf_acc, 4))
print("Macro-F1:", round(rbf_macro_f1, 4))
print("Weighted-F1:", round(rbf_weighted_f1, 4))
print(classification_report(y_test, y_pred_rbf, digits=4))

# 5) 混淆矩陣（觀察每一類的錯誤分佈）
cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)

print("\nConfusion Matrix - LinearSVC:\n", cm_linear)
print("\nConfusion Matrix - RBF SVC:\n", cm_rbf)

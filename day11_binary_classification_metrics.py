from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt

# 資料
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 模型
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, random_state=42))
])
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 評估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro Precision:", precision_score(y_test, y_pred, average="macro"))
print("Macro Recall:", recall_score(y_test, y_pred, average="macro"))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))

# 多分類 ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
print("Macro ROC-AUC:", roc_auc)

# 多分類 PR-AUC
pr_auc = average_precision_score(y_test, y_proba, average="macro")
print("Macro PR-AUC:", pr_auc)
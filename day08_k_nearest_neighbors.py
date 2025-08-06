import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# 資料產生
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2, n_redundant=0, 
    n_clusters_per_class=1, class_sep=1.2, random_state=42
)

# 資料分割與標準化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# for k in range(1, 21):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X_train_std, y_train, cv=5)
#     print(f"K={k}, Mean Accuracy={scores.mean():.3f}")

# 建模
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, y_train)

# 預測與評估
X_test_std = scaler.transform(X_test)
y_pred = knn.predict(X_test_std)
print(f"[Accuracy] {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 建立 mesh grid
h = .02
x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 畫圖
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
sns.scatterplot(x=X_train_std[:, 0], y=X_train_std[:, 1], hue=y_train, palette="Set1", edgecolor="k")
plt.title("KNN Decision Boundary (K=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

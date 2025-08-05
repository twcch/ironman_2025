from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 資料產生與前處理

X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)  # 標準化處理有助於收斂

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 訓練

model = LogisticRegression()
model.fit(X_train, y_train)

# 預測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 評估
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
print(classification_report(y_train, y_train_pred))

print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
print(classification_report(y_test, y_test_pred))

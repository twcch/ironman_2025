import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 資料產生與前處理
X, y = make_classification(n_samples=100000, n_features=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train_poly)

# 轉換為 tensor

X_train_tensor = torch.tensor(X_train_scaler, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

# 模型定義

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 1 個輸出 (sigmoid 之前)

    def forward(self, X):
        return self.sigmoid(self.linear(X))

model = LogisticRegressionModel(input_dim=X_train_poly.shape[1])

# 設定 criterion 與 optimizer

criterion = nn.BCELoss()

# weight_decay 是用來實作 L2 Regularization 的機制
# PyTorch 的 optim.SGD、optim.Adam 等 optimizer 本身沒有原生支援 L1 regularization 的核心原因在於以下三點:
## L1 不是 differentiable everywhere (在零點不可微)
## PyTorch 的 weight_decay 設計僅對應 L2 (即 Ridge)
## L1 實作需要手動處理梯度
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)

# 訓練

n_epochs = 1000
for epoch in range(n_epochs):
    model.train()  # 告訴模型進入: 訓練模式
    optimizer.zero_grad()  # 梯度歸零，預設梯度是累加的 (accumulated)，必須在每一回合開始前清空上一輪的梯度，否則梯度會疊加，導致錯誤的參數更新
    y_pred = model(X_train_tensor)  # 前向傳播 (forward pass)
    loss = criterion(y_pred, y_train_tensor)  # 使用定義好的損失函數來計算預測值和真實標籤之間的損失
    loss.backward()  # 反向傳播 (Backward Pass)，這些梯度會儲存在每個參數的 .grad 屬性中
    optimizer.step()  # 根據梯度更新模型參數
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 預測

X_test_poly = poly.fit_transform(X_test)
X_test_scaler = scaler.transform(X_test_poly)
X_test_tensor = torch.tensor(X_test_scaler, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

model.eval()
with torch.no_grad():  # 停用自動微分 (autograd)，不會追蹤張量的計算圖，減少記憶體消耗與加速推論，是測試與部署時的必備寫法，訓練需要追蹤梯度，測試不需要，這行就是明確告訴 PyTorch「這是測試」
    # 預測機率
    y_train_prob = model(X_train_tensor)
    y_test_prob = model(X_test_tensor)
    
    threshold = 0.5

    y_train_pred = (y_train_prob > threshold).int().numpy()
    y_test_pred = (y_test_prob > threshold).int().numpy()

# 評估
print(f"\n[Train Accuracy] {accuracy_score(y_train, y_train_pred):.2f}")
print(classification_report(y_train, y_train_pred))

print(f"\n[Test Accuracy] {accuracy_score(y_test, y_test_pred):.2f}")
print(classification_report(y_test, y_test_pred))

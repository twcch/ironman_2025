import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_boston_housing_data() -> pd.DataFrame:
    url = "http://lib.stat.cmu.edu/datasets/boston"  # Boston Housing Dataset from Carnegie Mellon University
    raw_data = requests.get(url).text.splitlines()[22:]  # 從第 23 行開始
    
    # 每筆資料分為 2 行 → 共 506 筆 → 總共 1012 行
    data = []
    for i in range(0, len(raw_data), 2):
        line_1 = list(map(float, raw_data[i].strip().split()))
        line_2 = list(map(float, raw_data[i + 1].strip().split()))
        data.append(line_1 + line_2)
    
    column_names = [
        "CRIM",  # 每人平均犯罪率
        "ZN",  # 區域住宅用地比例
        "INDUS",  # 區域非零售商業用地比例
        "CHAS",  # 查爾斯河虛擬變數 (1 = 河流旁, 0 = 其他)
        "NOX",  # 一氧化氮濃度 (parts per 10 million)
        "RM",  # 每個住宅的平均房間數
        "AGE",  # 1940 年之前建造的自用住宅比例
        "DIS",  # 到波士頓五個中心區域的加權距離
        "RAD",  # 公路接近指數 (1 = 最接近, 24 = 最遠)
        "TAX",  # 每 $10,000 的財產稅率
        "PTRATIO",  # 學生與教師比例
        "B",  # 1000(Bk - 0.63)^2, Bk = 區域黑人比例
        "LSTAT",  # 區域人口中低收入者的比例
        "MEDV",  # 自用住宅的中位數價格 (單位: $1000s)
    ]
    
    df = pd.DataFrame(data, columns=column_names)

    return df

def main():
    # ----- 讀取資料 -----
    
    original_data = get_boston_housing_data()  # 讀取資料
    
    # ----- 資料前處理 -----
    
    cleaned_data = original_data.copy()
    
    ## 因透過 print(cleaned_data.isnull().sum()) 檢查缺失值 (0 筆)
    ## 且 Pearson 相關係數檢查，相關係數 >= 0.8 (0 筆)    
    ### correlation_matrix = cleaned_data.corr()
    ### target_corr = correlation_matrix['MEDV']
    ### high_corr_features = target_corr[target_corr > 0.8].drop('MEDV')
    ### print(f"與 target 高度正相關的欄位:", high_corr_features)
    ## 故不做資料前處理
    
    # ----- 模型訓練 -----
    
    ## 資料分割
    X = cleaned_data.drop(columns=["MEDV"])
    y = cleaned_data["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ## 建立模型
    model = Pipeline([
        ("lr", LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    ## 預測
    y_pred = model.predict(X_test)
    
    ## 評估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"均方誤差 (MSE): {mse:.4f}")
    print(f"決定係數 R²: {r2:.4f}")
    
    ## 回歸係數檢視
    lr_model = model.named_steps["lr"]
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": lr_model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    
    print("各特徵回歸係數 (依影響力排序):")
    print(coefficients)
    
    ## 殘差分析
    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted MEDV")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

if __name__ == '__main__':
    main()

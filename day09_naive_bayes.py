import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 載入資料集
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 轉換目標變數為數值型
data['label_num'] = data.label.map({'ham':0, 'spam':1})

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# 特徵提取
vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

# 建立並訓練 Naive Bayes 模型
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# 預測
train_pred = nb.predict(X_train_dtm)
test_pred = nb.predict(X_test_dtm)

# 評估
print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(classification_report(y_train, train_pred))

print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
print(classification_report(y_test, test_pred))
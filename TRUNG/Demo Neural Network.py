import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Đọc dữ liệu
df = pd.read_csv('./badminton_dataset.csv')

# Mã hóa các thuộc tính categorical
le = LabelEncoder()
outlook = le.fit_transform(df['Outlook'].values)
temperature = le.fit_transform(df['Temperature'].values)
humidity = le.fit_transform(df['Humidity'].values)
wind = le.fit_transform(df['Wind'].values)
play_badminton = le.fit_transform(df['Play_Badminton'].values)

# Chuẩn bị dữ liệu đầu vào
X_data = np.array([outlook, temperature, humidity, wind]).T
y_data = play_badminton

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=False)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network)
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam')

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# In kết quả dự đoán so với thực tế
print("Thực tế \t Dự đoán")
for i in range(len(y_test)):
    print(y_test[i], "\t\t", y_pred[i])

# Đánh giá mô hình
print("Độ đo precision:", precision_score(y_test, y_pred, average='micro'))
print("Độ đo recall:", recall_score(y_test, y_pred, average='micro'))
print("Độ đo F1:", f1_score(y_test, y_pred, average='micro'))

# Tỷ lệ dự đoán đúng
accuracy = clf.score(X_test, y_test)
print('Tỷ lệ dự đoán đúng:', np.around(accuracy * 100, 2), '%')

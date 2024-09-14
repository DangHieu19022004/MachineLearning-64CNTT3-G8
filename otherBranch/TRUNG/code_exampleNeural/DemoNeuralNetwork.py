import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ seattle-weather.csv
df = pd.read_csv("../seattle-weather.csv", index_col=0)

# Xem xét các giá trị của các cột
print(df.head())

# Mã hóa các thuộc tính categorical
le = LabelEncoder()

# Mã hóa các thuộc tính thời tiết và các đặc tính liên quan
precipitation = le.fit_transform(df['precipitation'].values)
temp_max = le.fit_transform(df['temp_max'].values)
temp_min = le.fit_transform(df['temp_min'].values)
wind = le.fit_transform(df['wind'].values)
weather = le.fit_transform(df['weather'].values)  # Cột nhãn (target)

# Chuẩn bị dữ liệu đầu vào
X_data = np.array([precipitation, temp_max, temp_min, wind]).T
y_data = weather

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


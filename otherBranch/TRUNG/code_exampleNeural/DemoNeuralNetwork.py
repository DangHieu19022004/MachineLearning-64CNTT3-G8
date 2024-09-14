import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ seattle-weather.csv
df = pd.read_csv("../seattle-weather.csv", index_col=0)

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với MLP
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Nhãn (target)
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra (shuffle=True để trộn ngẫu nhiên dữ liệu)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network)
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# In kết quả dự đoán so với thực tế (in 10 dòng để dễ đọc)
print("Thực tế \t Dự đoán")
for i in range(10):
    print(le.inverse_transform([y_test[i]])[0], "\t\t", le.inverse_transform([y_pred[i]])[0])

# Đánh giá mô hình và in bảng phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# In độ chính xác (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print('Tỷ lệ dự đoán đúng (accuracy):', np.around(accuracy * 100, 2), '%')

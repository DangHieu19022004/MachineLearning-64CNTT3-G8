import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Đảm bảo hệ thống sử dụng mã hóa UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Đọc dữ liệu từ file CSV
data = pd.read_csv('F:/OneDrive/Máy tính/BTL/BTL/seattle-weather.csv')

# Kiểm tra và xử lý giá trị thiếu
data = data.dropna()

# Kiểm tra phân phối lớp
print("Phân phối lớp:")
print(data['weather'].value_counts())

# Tiền xử lý dữ liệu
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = data['weather']

# Chuyển đổi nhãn lớp thành số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình MLP (Multilayer Perceptron) không sử dụng class_weight
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# In kết quả dự đoán so với thực tế (in 10 dòng để dễ đọc)
print("Thực tế \t Dự đoán")
for i in range(10):
    print(label_encoder.inverse_transform([y_test[i]])[0], "\t\t", label_encoder.inverse_transform([y_pred[i]])[0])

# Đánh giá mô hình và in báo cáo phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

# In độ chính xác (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print('Tỷ lệ dự đoán đúng (accuracy):', np.around(accuracy * 100, 2), '%')

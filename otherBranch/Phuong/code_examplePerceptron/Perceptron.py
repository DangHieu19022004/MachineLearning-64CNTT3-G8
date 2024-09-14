import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc tệp dữ liệu
data = pd.read_csv("../../../seattle-weather.csv", index_col=0)

# Sử dụng các cột: precipitation, temp_max, temp_min, wind
features = ["precipitation", "temp_max", "temp_min", "wind"]
X = data[features]

# Sử dụng LabelEncoder để mã hóa cột 'weather'
le = LabelEncoder()
y = le.fit_transform(data["weather"])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False)

# Khởi tạo và huấn luyện mô hình Perceptron
clf = Perceptron()
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_valid)

# Hiển thị kết quả thực tế và dự đoán dưới dạng số
print("Thực tế \t Dự đoán")
for true, pred in zip(y_valid, y_pred):
    print(true, "\t\t", pred)

# Tính toán tỷ lệ dự đoán chính xác
accuracy = np.mean(y_valid == y_pred) * 100
print(f'Tỷ lệ dự đoán đúng: {accuracy:.2f} %')

# Tính toán precision, recall và F1-score
print("Độ đo precision:", precision_score(y_valid, y_pred, average='micro'))
print("Độ đo recall:", recall_score(y_valid, y_pred, average='micro'))
print("Độ đo F1:", f1_score(y_valid, y_pred, average='micro'))

# Hiển thị ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Dự đoán với dữ liệu mới
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})
new_prediction = clf.predict(new_input)
print("Dự đoán thời tiết cho dữ liệu mới:", new_prediction)

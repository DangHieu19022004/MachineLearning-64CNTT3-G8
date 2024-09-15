import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Đọc tệp dữ liệu
data = pd.read_csv(r'C:\Users\LENOVO\Downloads\Học máy\seattle-weather.csv', index_col=0)  

# Sử dụng tất cả bốn thuộc tính
features = ["precipitation", "temp_max", "temp_min", "wind"]
X = data[features]

# Mã hóa cột 'weather' thành số
le = LabelEncoder()
y = le.fit_transform(data["weather"])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, shuffle=False)

# Khởi tạo và huấn luyện mô hình Perceptron
clf = Perceptron()
clf.fit(X_train, y_train)

# Giảm chiều dữ liệu từ 4 chiều xuống 2 chiều với PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Chia dữ liệu đã giảm chiều
X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(X_pca, y, train_size=0.7, test_size=0.3, shuffle=False)

# Huấn luyện lại mô hình với dữ liệu đã giảm chiều
clf.fit(X_train_pca, y_train_pca)

# Tạo lưới điểm cho 2D
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Dự đoán lớp cho mỗi điểm trên lưới (không sử dụng PCA để dự đoán)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ decision boundary với dữ liệu đã giảm chiều
plt.figure(figsize=(12, 6))

plt.subplot(1, 1, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Boundary với PCA')

plt.tight_layout()
plt.show()

# Dự đoán trên tập kiểm tra đã giảm chiều
y_pred_pca = clf.predict(X_valid_pca)

# Tính toán tỷ lệ dự đoán chính xác
accuracy_pca = np.mean(y_valid_pca == y_pred_pca) * 100
print(f'Tỷ lệ dự đoán đúng (PCA): {accuracy_pca:.2f} %')

# Tính toán precision, recall và F1-score
print("Độ đo precision (PCA):", precision_score(y_valid_pca, y_pred_pca, average='micro', zero_division=1))
print("Độ đo recall (PCA):", recall_score(y_valid_pca, y_pred_pca, average='micro'))
print("Độ đo F1 (PCA):", f1_score(y_valid_pca, y_pred_pca, average='micro'))

# In báo cáo phân loại (Classification Report)
report_pca = classification_report(y_valid_pca, y_pred_pca, target_names=le.classes_, zero_division=1)
print("Báo cáo phân loại (PCA):\n", report_pca)

# Hiển thị ma trận nhầm lẫn (Confusion Matrix) cho dữ liệu đã giảm chiều
cm_pca = confusion_matrix(y_valid_pca, y_pred_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=le.classes_)
disp_pca.plot(cmap=plt.cm.Blues)
plt.show()

# Dự đoán với dữ liệu mới
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})

# Áp dụng PCA cho dữ liệu mới trước khi dự đoán
new_input_pca = pca.transform(new_input)
new_prediction = clf.predict(new_input_pca)
predicted_weather = new_prediction
print("Dự đoán thời tiết cho dữ liệu mới:", predicted_weather[0])

import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc tệp dữ liệu
data = pd.read_csv("./seattle-weather.csv", index_col=0)

# Sử dụng tất cả bốn thuộc tính
features = ["precipitation", "temp_max", "temp_min", "wind"]
X = data[features]

# Mã hóa cột 'weather' thành số
le = LabelEncoder()
y = le.fit_transform(data["weather"])
# Giảm chiều dữ liệu từ 4 chiều xuống 2 chiều với PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Chia dữ liệu đã giảm chiều
X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(X_pca, y, train_size=0.7, test_size=0.3, shuffle=False)

# Huấn luyện mô hình Perceptron với dữ liệu đã giảm chiều
clf = Perceptron()
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
plt.figure(figsize=(10, 6))

plt.subplot(1, 1, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Boundary với PCA')
plt.tight_layout()
    #Save the plot to a BytesIO object
imgPCA = io.BytesIO()
plt.savefig(imgPCA, format='png')
imgPCA.seek(0)
    #Encode image to base64
plot_url_PCA = base64.b64encode(imgPCA.getvalue()).decode()


# Dự đoán trên tập kiểm tra đã giảm chiều
y_pred_pca = clf.predict(X_valid_pca)


# Tính toán tỷ lệ dự đoán chính xác
accuracy_pca = np.mean(y_valid_pca == y_pred_pca) * 100

# Tính toán precision, recall và F1-score
# print("Độ đo precision (PCA):", precision_score(y_valid_pca, y_pred_pca, average='micro', zero_division=1))
# print("Độ đo recall (PCA):", recall_score(y_valid_pca, y_pred_pca, average='micro'))
# print("Độ đo F1 (PCA):", f1_score(y_valid_pca, y_pred_pca, average='micro'))

# In báo cáo phân loại (Classification Report)
report_pca = classification_report(y_valid_pca, y_pred_pca, target_names=le.classes_, zero_division=1)

# Hiển thị ma trận nhầm lẫn (Confusion Matrix) cho dữ liệu đã giảm chiều
cm_pca = confusion_matrix(y_valid_pca, y_pred_pca)
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp_pca.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix (PCA)")
    #Save the plot to a BytesIO object
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
    #Encode image to base64
plot_url = base64.b64encode(img.getvalue()).decode()


# Dự đoán với dữ liệu mới
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})


def predict_weather(new_input):
    new_input_pca = pca.transform(new_input)
    predicted_encoded = clf.predict(new_input_pca)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels


# # Áp dụng PCA cho dữ liệu mới trước khi dự đoán
# predicted_weather = predict_weather(new_input)
# print("Dự đoán thời tiết cho dữ liệu mới:", predicted_weather.tolist())



valueSend = {
    'model': clf,
    'accuracy': accuracy_pca,
    'report': report_pca,
    'plot_url': plot_url,
    'entropy_url': plot_url_PCA
}

# Save the model
joblib.dump(valueSend, 'perceptron_model.pkl')

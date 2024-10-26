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
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc tệp dữ liệu
data = pd.read_csv("./seattle-weather.csv")

# Lọc dữ liệu
data = data.dropna()

# Các cột đặc trưng để huấn luyện mô hình
features = ["precipitation", "temp_max", "temp_min", "wind"]

# Chia dữ liệu thành X và y
X = data[features]
y = data["weather"]

# Mã hóa nhãn
le = LabelEncoder()
y = le.fit_transform(y)


# Chia dữ liệu thành 70% huấn luyện, 15% xác thực, 15% kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Khởi tạo và huấn luyện mô hình Perceptron
perceptron_model = Perceptron(max_iter=1000, random_state=42, eta0=0.01)
perceptron_model.fit(X_train, y_train)

# Dự đoán trên các tập dữ liệu
y_train_pred = perceptron_model.predict(X_train)
y_valid_pred = perceptron_model.predict(X_valid)
y_test_pred = perceptron_model.predict(X_test)

# Báo cáo phân loại
report_validation = classification_report(y_valid, y_valid_pred, target_names=le.classes_, zero_division=0)
report_trainning_set = classification_report(y_train, y_train_pred, target_names=le.classes_, zero_division=0)
report_test_set = classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0)

print("Báo cáo cho tập xác thực:")
print(report_validation)
print("Báo cáo cho tập huấn luyện:")
print(report_trainning_set)
print("Báo cáo cho tập kiểm tra:")
print(report_test_set)

# Ma trận nhầm lẫn cho tập kiểm tra
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
plot_url = base64.b64encode(img.getvalue()).decode()

# Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    perceptron_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid(True)
imgLearningCurve = io.BytesIO()
plt.savefig(imgLearningCurve, format='png')
imgLearningCurve.seek(0)
learning_curve_url = base64.b64encode(imgLearningCurve.getvalue()).decode()
# plt.show()

# # Dự đoán trên tập kiểm tra đã giảm chiều
# y_pred_pca = clf.predict(X_valid_pca)


# # Tính toán tỷ lệ dự đoán chính xác
# accuracy_pca = np.mean(y_valid_pca == y_pred_pca) * 100

# Tính toán precision, recall và F1-score
# print("Độ đo precision (PCA):", precision_score(y_valid_pca, y_pred_pca, average='micro', zero_division=1))
# print("Độ đo recall (PCA):", recall_score(y_valid_pca, y_pred_pca, average='micro'))
# print("Độ đo F1 (PCA):", f1_score(y_valid_pca, y_pred_pca, average='micro'))

# # In báo cáo phân loại (Classification Report)
# report_pca = classification_report(y_valid_pca, y_pred_pca, target_names=le.classes_, zero_division=1)

# # Hiển thị ma trận nhầm lẫn (Confusion Matrix) cho dữ liệu đã giảm chiều
# cm_pca = confusion_matrix(y_valid_pca, y_pred_pca)
# disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=le.classes_)
# fig, ax = plt.subplots(figsize=(10, 7))
# disp_pca.plot(cmap=plt.cm.Blues, ax=ax)
# plt.title("Confusion Matrix (PCA)")
#     #Save the plot to a BytesIO object
# img = io.BytesIO()
# plt.savefig(img, format='png')
# img.seek(0)
#     #Encode image to base64
# plot_url = base64.b64encode(img.getvalue()).decode()


# Dự đoán với dữ liệu mới
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})


def predict_weather(new_input):
    # new_input_pca = pca.transform(new_input)
    predicted_encoded = perceptron_model.predict(new_input)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels


# # Áp dụng PCA cho dữ liệu mới trước khi dự đoán
# predicted_weather = predict_weather(new_input)
# print("Dự đoán thời tiết cho dữ liệu mới:", predicted_weather.tolist())



# Lưu mô hình và kết quả
valueSend = {
    'model': perceptron_model,
    'report_validation': report_validation,
    'report_trainning_set': report_trainning_set,
    'report_test_set': report_test_set,
    'plot_url': plot_url,
    'learning_curve_url': learning_curve_url
}

# Save the model
joblib.dump(valueSend, 'perceptron_model.pkl')

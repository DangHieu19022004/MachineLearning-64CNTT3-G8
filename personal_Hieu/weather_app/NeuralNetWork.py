import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# Đọc dữ liệu từ file CSV
df = pd.read_csv("../../seattle-weather.csv")

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

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra (shuffle=True để trộn ngẫu nhiên dữ liệu)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network)
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Huấn luyện mô hình
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})

def encode_input(new_input):
    predicted_encoded = clf.predict(new_input)
    predicted_labels = le.inverse_transform(predicted_encoded)
    return predicted_labels

# Đánh giá mô hình và in bảng phân loại chi tiết
# # In độ chính xác (accuracy)
accuracy = accuracy_score(y_test, y_pred) * 100
#
# In ma trận nhầm lẫn (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)

# 1. Vẽ ma trận nhầm lẫn (Confusion Matrix)
fig, ax = plt.subplots(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('True label')  # Nhãn thực tế
plt.xlabel('Predicted label')  # Nhãn dự đoán
plt.title('Confusion Matrix')  # Tiêu đề: Ma trận nhầm lẫn

    #Save the plot to a BytesIO object
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
    #Encode image to base64
plot_url = base64.b64encode(img.getvalue()).decode()

#báo cáo phân loại
report = classification_report(y_test, y_pred, target_names=le.classes_,zero_division=0)

# 3. Vẽ đường cong ROC cho từng lớp (ROC Curve)
# Binarize (mã hóa nhị phân) nhãn đầu ra cho phân loại đa lớp ROC
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
y_pred_bin = label_binarize(y_pred, classes=np.arange(len(le.classes_)))

# Vẽ đường cong ROC cho từng lớp
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])  # Tính FPR và TPR cho lớp i
    roc_auc = auc(fpr, tpr)  # Tính AUC cho lớp i
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')  # Vẽ ROC cho từng lớp

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Đường tham chiếu ROC
plt.xlim([0.0, 1.0])  # Giới hạn trục X
plt.ylim([0.0, 1.05])  # Giới hạn trục Y
plt.xlabel('False Positive Rate')  # Trục X: Tỷ lệ dương tính giả
plt.ylabel('True Positive Rate')  # Trục Y: Tỷ lệ dương tính thực
plt.title('ROC Curve for Multi-Class Classification')  # Tiêu đề: Đường cong ROC cho phân loại đa lớp
plt.legend(loc='lower right')  # Hiển thị chú thích ở góc phải dưới
    #Save the plot to a BytesIO object
imgROC = io.BytesIO()
plt.savefig(imgROC, format='png')
imgROC.seek(0)
    #Encode image to base64
ROC_url = base64.b64encode(imgROC.getvalue()).decode()

valueSend = {
    'model': clf,
    'accuracy': accuracy,
    'report': report,
    'plot_url': plot_url,
    'entropy_url': ROC_url
}

# Save the model
joblib.dump(valueSend, 'neural_network_model.pkl')

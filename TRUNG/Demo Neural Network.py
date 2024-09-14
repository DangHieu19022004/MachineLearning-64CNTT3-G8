import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./seattle-weather.csv')

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

# Đánh giá mô hình và in bảng phân loại chi tiết
print("\nBáo cáo phân loại chi tiết:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# In độ chính xác (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print('Tỷ lệ dự đoán đúng (accuracy):', np.around(accuracy * 100, 2), '%')

# In ma trận nhầm lẫn (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMa trận nhầm lẫn (Confusion Matrix):\n")
print(conf_matrix)

# 1. Vẽ ma trận nhầm lẫn (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('True label')  # Nhãn thực tế
plt.xlabel('Predicted label')  # Nhãn dự đoán
plt.title('Confusion Matrix')  # Tiêu đề: Ma trận nhầm lẫn
plt.show()

# 2. Vẽ biểu đồ heatmap của báo cáo phân loại (Classification Report)
# Tạo báo cáo phân loại và chuyển đổi thành DataFrame
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Vẽ heatmap của các chỉ số báo cáo phân loại
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report Heatmap')  # Tiêu đề: Báo cáo phân loại dưới dạng heatmap
plt.show()

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
plt.show()
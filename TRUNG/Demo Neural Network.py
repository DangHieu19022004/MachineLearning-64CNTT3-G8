import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Đọc dữ liệu từ file CSV
df = pd.read_csv('./seattle-weather.csv')

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Lưu LabelEncoder để sử dụng trong quá trình deploy lên web
joblib.dump(le, 'label_encoder.pkl')

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với MLP
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Lưu scaler để sử dụng trong quá trình deploy lên web
joblib.dump(scaler, 'scaler.pkl')

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành 3 tập: training (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Lấy một phần dữ liệu huấn luyện (giảm kích thước)
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Xây dựng mô hình MLP (Multilayer Perceptron - Neural Network) với Early Stopping
clf = MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=500, activation='relu', solver='adam', random_state=42, early_stopping=True, learning_rate_init=0.01)

# Huấn luyện mô hình
clf.fit(X_train_sample, y_train_sample)

# Lưu mô hình để triển khai sau này
joblib.dump(clf, 'neural_network_model.pkl')

# Dự đoán trên các tập dữ liệu
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

# Đánh giá mô hình trên cả 3 tập
train_report = classification_report(y_train, y_train_pred, target_names=le.classes_, zero_division=1)
val_report = classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=1)
test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=1)

# Tổng hợp báo cáo
report = {
    'train': train_report,
    'validation': val_report,
    'test': test_report
}

# Vẽ ma trận nhầm lẫn (Confusion Matrix) cho tập test
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    
    # Lưu ma trận nhầm lẫn vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Lấy ảnh base64 của Confusion Matrix cho Test Set
confusion_matrix_base64 = plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix - Test Set')

# Vẽ biểu đồ Learning Curve
def plot_learning_curve(estimator, X, y, title, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 3)):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")

    # Lấy giá trị training và validation scores cho từng kích thước tập huấn luyện
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Tính trung bình và độ lệch chuẩn
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Vẽ biểu đồ training và validation scores
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation score")

    plt.legend(loc="best")
    
    # Lưu Learning Curve vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Lấy ảnh base64 của Learning Curve
learning_curve_base64 = plot_learning_curve(clf, X_train_sample, y_train_sample, "Learning Curve - MLP Classifier", cv=5)

# Giá trị gửi đi bao gồm các đường dẫn ảnh base64 của ma trận nhầm lẫn và Learning Curve
valueSend = {
    'model': 'neural_network_model.pkl',
    'report': report,
    'confusion_matrix_base64': confusion_matrix_base64,
    'learning_curve_base64': learning_curve_base64
}
print(valueSend)


'''
valueSend_short = valueSend.copy()
valueSend_short.pop('confusion_matrix_base64')
valueSend_short.pop('learning_curve_base64')

print(json.dumps(valueSend_short, indent=4))
'''










#Learning curve






'''
# Vẽ hàm mất mát (Loss Function) trong quá trình huấn luyện
plt.figure(figsize=(8, 6))
plt.plot(clf.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid()
plt.show()
'''


'''
# 2. Vẽ biểu đồ heatmap của báo cáo phân loại (Classification Report)
# Tạo báo cáo phân loại và chuyển đổi thành DataFrame (thêm zero_division=1)
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=1)
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
'''

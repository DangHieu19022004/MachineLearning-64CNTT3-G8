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
                             recall_score, log_loss)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Đọc tệp dữ liệu
df = pd.read_csv(r'C:\Users\LENOVO\Downloads\Học máy\seattle-weather.csv', index_col=0)  

# Xử lý dữ liệu thiếu (nếu có) bằng cách loại bỏ các hàng có giá trị null
df = df.dropna()

# Mã hóa thuộc tính weather (categorical) thành số
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

# Lưu LabelEncoder để sử dụng trong quá trình triển khai
joblib.dump(le, 'label_encoder.pkl')

# Sử dụng các cột numerical như precipitation, temp_max, temp_min, wind làm dữ liệu đầu vào
X_data = df[['precipitation', 'temp_max', 'temp_min', 'wind']].values

# Chuẩn hóa (scale) các thuộc tính numerical để phù hợp với Perceptron
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)

# Lưu scaler để sử dụng trong quá trình triển khai
joblib.dump(scaler, 'scaler.pkl')

# Nhãn (target) là cột weather đã được mã hóa
y_data = df['weather_encoded'].values

# Chia tập dữ liệu thành 3 tập: training (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Xây dựng và huấn luyện mô hình Perceptron
clf = Perceptron(max_iter=1000, random_state=42, eta0=0.01)

# Vẽ lại hàm mất mát (Loss Function) cho Perceptron
# Khởi tạo danh sách để lưu trữ mất mát và độ chính xác
loss_history = []
accuracy_history = []

# Số lượng epochs 
n_epochs = 700

# Lấy giá trị lớp duy nhất cho partial_fit
classes = np.unique(y_train)

# Vòng lặp qua các epochs để huấn luyện tăng dần và theo dõi mất mát
for epoch in range(n_epochs):
    clf.partial_fit(X_train, y_train, classes=classes)
    
    # Dự đoán trên tập huấn luyện
    y_train_pred = clf.predict(X_train)
    
    # Tính toán mất mát hiện tại
    current_loss = log_loss(y_train, clf.decision_function(X_train))
    loss_history.append(current_loss)
    
    # Tính toán độ chính xác trên tập huấn luyện
    train_accuracy = np.mean(y_train_pred == y_train)
    accuracy_history.append(train_accuracy)
    
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {current_loss:.4f} - Accuracy: {train_accuracy:.4f}")

# Lưu mô hình để triển khai sau này
joblib.dump(clf, 'perceptron_model.pkl')

# Sau khi huấn luyện, đánh giá trên tập test
y_test_pred = clf.predict(X_test)

# Dự đoán trên các tập dữ liệu
y_train_pred = clf.predict(X_train)
y_val_pred = clf.predict(X_val)

# Hàm đánh giá mô hình và in báo cáo
def evaluate_model(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=1)
    print(f"\n{title} Classification Report:")
    print(report)
    return report

# Đánh giá mô hình trên cả 3 tập
train_report = evaluate_model(y_train, y_train_pred, "Training")
val_report = evaluate_model(y_val, y_val_pred, "Validation")
test_report = evaluate_model(y_test, y_test_pred, "Test")

# Vẽ biểu đồ mất mát
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), loss_history, marker='o', label='Training Loss')
plt.title('Perceptron Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Lưu biểu đồ mất mát vào bộ nhớ dưới dạng ảnh base64
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png')
img_buffer.seek(0)
loss_curve_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
plt.close()

# Vẽ biểu đồ độ chính xác
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), accuracy_history, marker='o', color='g', label='Training Accuracy')
plt.title('Perceptron Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Hàm vẽ biểu đồ Confusion Matrix cho tập test
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.title(title)
    plt.show()
    
    # Lưu ma trận nhầm lẫn vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Lấy ảnh base64 của Confusion Matrix cho Test Set
confusion_matrix_base64 = plot_confusion_matrix(y_test, y_test_pred, 'Ma trận nhầm lẫn - Tập Test')

# Hàm vẽ biểu đồ Learning Curve với các chi tiết bổ sung
def plot_learning_curve_v2(estimator, X, y, title, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=14)
    plt.xlabel("Số lượng mẫu huấn luyện", fontsize=12)
    plt.ylabel("Điểm số", fontsize=12)

    # Lấy giá trị training và validation scores cho từng kích thước tập huấn luyện
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Tính trung bình và độ lệch chuẩn
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Vẽ biểu đồ training và validation scores với các chi tiết bổ sung
    plt.grid(True)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Validation score")

    plt.legend(loc="best")
    plt.show()

    # Lưu Learning Curve vào bộ nhớ dưới dạng ảnh base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    learning_curve_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return learning_curve_base64

# Lấy lại Learning Curve base64 cho Perceptron
learning_curve_base64 = plot_learning_curve_v2(clf, X_train, y_train, "Learning Curve - Perceptron Classifier", cv=5)

# Giá trị gửi đi bao gồm các đường dẫn ảnh base64 của ma trận nhầm lẫn, Learning Curve và Loss Curve
valueSend = {
    'model': 'perceptron_model.pkl',
    'report': {
        'train': train_report,
        'validation': val_report,
        'test': test_report,
    },
    'images': {
        'loss_curve': loss_curve_base64,
        'confusion_matrix': confusion_matrix_base64,
        'learning_curve': learning_curve_base64,
    }
}


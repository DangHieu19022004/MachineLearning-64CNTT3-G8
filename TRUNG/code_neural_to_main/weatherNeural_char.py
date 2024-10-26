import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve  

# Nạp mô hình và dữ liệu
best_nn_model = joblib.load('best_nn_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
X_train, X_test, y_train, y_test = joblib.load('data_splits.pkl')

# Dự đoán trên tập kiểm tra
y_test_pred_nn = best_nn_model.predict(X_test)

# 1. Ma trận nhầm lẫn
cm_nn = confusion_matrix(y_test, y_test_pred_nn)
disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp_nn.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('Nhãn Thực')
plt.xlabel('Nhãn Dự Đoán')
plt.title("Ma trận nhầm lẫn cho Mạng Nơ-ron trên Tập Kiểm Tra")
plt.show()

# 2. Đường cong học tập (Learning Curve)
train_sizes, train_scores, val_scores = learning_curve(
    best_nn_model, X_train, y_train, cv=5, scoring='accuracy', 
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

# Tính giá trị trung bình và độ lệch chuẩn cho điểm số huấn luyện và xác thực
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Tính điểm số trên tập test cho mỗi kích thước tập huấn luyện
test_scores = []
for size in train_sizes:
    # Chia dữ liệu theo kích thước tập huấn luyện
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size/X_train.shape[0], random_state=42)
    
    # Huấn luyện mô hình với tập con và dự đoán trên tập kiểm tra
    best_nn_model.fit(X_train_subset, y_train_subset)
    y_test_pred = best_nn_model.predict(X_test)
    
    # Đánh giá điểm số trên tập kiểm tra
    test_scores.append(accuracy_score(y_test, y_test_pred))

test_scores = np.array(test_scores)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label="Validation score")
plt.plot(train_sizes, test_scores, 'o-', color='b', label="Test score")

# Thêm các vùng che phủ (confidence interval) cho điểm số huấn luyện và xác thực
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='g')

# Thiết lập tiêu đề và nhãn
plt.title("Learning Curve (Neural Network)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid()
plt.show()

# Huấn luyện lại mô hình để lưu trữ loss curve
best_nn_model.fit(X_train, y_train)

# Vẽ biểu đồ hàm mất mát (Loss Curve)
plt.figure(figsize=(10, 6))
plt.plot(best_nn_model.loss_curve_, label='Training Loss', color='purple')
plt.title("Loss Curve for Neural Network")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.grid()
plt.show()

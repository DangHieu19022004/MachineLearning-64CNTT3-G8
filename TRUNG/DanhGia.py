import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu và xử lý
df = pd.read_csv('./seattle-weather.csv').dropna()
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])
joblib.dump(le, 'label_encoder.pkl')

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_data = scaler.fit_transform(df[['precipitation', 'temp_max', 'temp_min', 'wind']].values)
joblib.dump(scaler, 'scaler.pkl')  # Lưu scaler đã được huấn luyện
y_data = df['weather_encoded'].values

# Chia dữ liệu thành các tập
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Khám phá ảnh hưởng của số lượng neuron trong các lớp ẩn
hidden_layer_sizes_options = [(10,), (20,), (30, 15), (40, 20, 10)]  # Các cấu hình lớp ẩn để thử nghiệm
results = []

for hidden_layer_sizes in hidden_layer_sizes_options:
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, activation='relu', solver='adam',
                        random_state=42, early_stopping=True, learning_rate_init=0.01)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    mean_cv_score = round(np.mean(cv_scores), 4)  # Làm tròn đến 4 chữ số thập phân
    results.append((hidden_layer_sizes, mean_cv_score))
    print(f"Hidden layer sizes: {hidden_layer_sizes}, Cross-validation score: {mean_cv_score}")

# Lưu trữ kết quả vào DataFrame để dễ dàng hiển thị
results_df = pd.DataFrame(results, columns=['Hidden Layer Sizes', 'Mean Cross-validation Score'])

# Đảm bảo rằng các giá trị trong 'Hidden Layer Sizes' là chuỗi
results_df['Hidden Layer Sizes'] = results_df['Hidden Layer Sizes'].astype(str)

# Huấn luyện mô hình với cấu hình tốt nhất
best_hidden_layer_sizes = results_df.loc[results_df['Mean Cross-validation Score'].idxmax(), 'Hidden Layer Sizes']
# Chuyển đổi về tuple
best_hidden_layer_sizes = eval(best_hidden_layer_sizes)  
print(f"Cấu hình lớp ẩn tốt nhất: {best_hidden_layer_sizes}")

# Huấn luyện và lưu mô hình
best_clf = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, max_iter=500, activation='relu', solver='adam',
                          random_state=42, early_stopping=True, learning_rate_init=0.01)
best_clf.fit(X_train, y_train)
joblib.dump(best_clf, 'best_neural_network_model.pkl')

# Dự đoán và báo cáo
for (X, y, name) in [(X_train, y_train, 'Training'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]:
    print(f"\nClassification report for {name} set:\n", classification_report(y, best_clf.predict(X), target_names=le.classes_, zero_division=1))

# Hàm vẽ và lưu ma trận nhầm lẫn
def save_plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.close()


# Hàm vẽ Learning Curve
def save_plot_learning_curve(estimator, X, y, cv, filename):
    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve - MLP Classifier")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5))
    plt.grid()
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color="r")
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color="g")
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.close()


# Vẽ sơ đồ phân tán trước khi chuẩn hóa
sns.pairplot(df[['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']], hue='weather')
plt.suptitle("Sơ đồ phân tán trước khi chuẩn hóa", y=1.02)
plt.tight_layout()

plt.show()

# Vẽ sơ đồ phân tán sau khi chuẩn hóa
df_scaled = pd.DataFrame(X_data, columns=['precipitation', 'temp_max', 'temp_min', 'wind'])
df_scaled['weather'] = df['weather']
sns.pairplot(df_scaled, hue='weather')
plt.suptitle("Sơ đồ phân tán sau khi chuẩn hóa", y=1.02)
plt.tight_layout()
plt.show()


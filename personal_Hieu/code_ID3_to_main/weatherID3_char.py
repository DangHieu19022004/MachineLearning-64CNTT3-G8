#predict weather based on dataset

#import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("../weather_app/seattle-weather.csv")

# #filter data
data = data.dropna()
data.drop(['date'], axis=1, inplace=True)

#Split the datasets into X and y
X = data[["precipitation", "temp_max", "temp_min", "wind"]]
y = data["weather"]

#Split the datasets into 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


best_model = DecisionTreeClassifier(criterion='entropy', random_state=42,
                                    class_weight = None,
                                    max_depth = 3,
                                    max_features = 'sqrt',
                                    max_leaf_nodes = None,
                                    min_samples_leaf = 1
                                    , min_samples_split = 2,
                                    min_weight_fraction_leaf = 0.0,
                                    splitter = 'best')

best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

# tính toán độ chính xác mô hình
accuracy = accuracy_score(y_test, y_test_pred) * 100


# 2. Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix on test set")


# 3. In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
# 3.1. report training set
report_trainning_set = classification_report(y_train, y_train_pred, target_names=best_model.classes_, zero_division=0)
# 3.2. report validation set
report_validation = classification_report(y_valid, y_valid_pred, target_names=best_model.classes_, zero_division=0)
# 3.3. report test set
report_test_set = classification_report(y_test, y_test_pred, target_names=best_model.classes_, zero_division=0)

print("trainning")
print(report_trainning_set)
print("vali")
print(report_validation)
print("test")
print(report_test_set)


# 4. Vẽ sơ đồ learning curve cho cả tập huấn luyện, xác thực và kiểm tra
# Initialize the KFold cross-validator with desired settings, e.g., 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, cv=kf, scoring='accuracy',
    n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

# Tính toán điểm trung bình và khoảng tin cậy
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)


# Tính điểm accuracy cho tập test ở từng kích thước của train_sizes
test_scores = []
for train_size in train_sizes:
    # Tạo tập con của dữ liệu huấn luyện
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    best_model.fit(X_train_subset, y_train_subset)
    test_scores.append(accuracy_score(y_test, best_model.predict(X_test)))

# Chuyển đổi test_scores sang numpy array
test_scores = np.array(test_scores)

# Vẽ learning curve
plt.figure()
plt.title('Learning Curve (Decision Tree)')
plt.xlabel('Training examples')
plt.ylabel('Score')

# Trực quan hóa khoảng tin cậy của tập huấn luyện và xác thực
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

# Vẽ các đường learning curve
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
plt.plot(train_sizes, test_scores, 'o-', color="b", label="Test score")

# Hiển thị legend và biểu đồ
plt.legend(loc="best")
plt.show()



# 5. Vẽ biểu đồ ROC và tính AUC cho phân loại đa lớp
# Chuyển đổi nhãn sang định dạng one-hot encoding
# Lấy danh sách các lớp
classes = best_model.classes_
y_valid_bin = label_binarize(y_valid, classes=classes)
n_classes = y_valid_bin.shape[1]

# Tính xác suất dự đoán cho mỗi lớp
y_prob = best_model.predict_proba(X_valid)

# Tính toán ROC curve và AUC cho mỗi lớp
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_valid_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Vẽ ROC cho từng lớp
plt.figure()

colors = plt.cm.get_cmap('tab10', n_classes)(range(n_classes))

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc="lower right")
plt.show()

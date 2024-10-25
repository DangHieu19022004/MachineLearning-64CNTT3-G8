#predict weather based on dataset

#import libraries
import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, learning_curve,
                                     train_test_split)
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

# In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
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

# # 4. Vẽ sơ đồ learning curve cho cả tập huấn luyện, xác thực và kiểm tra
# train_sizes, train_scores, valid_scores = learning_curve(
#     best_model, X_train, y_train, cv=kf, scoring='accuracy',
#     n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
# )

# # Tính toán điểm trung bình và khoảng tin cậy
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# valid_scores_mean = np.mean(valid_scores, axis=1)
# valid_scores_std = np.std(valid_scores, axis=1)

# # Tính điểm accuracy cho tập test ở từng kích thước của train_sizes
# test_scores = []
# for train_size in train_sizes:
#     # Tạo tập con của dữ liệu huấn luyện
#     X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
#     best_model.fit(X_train_subset, y_train_subset)
#     test_scores.append(accuracy_score(y_test, best_model.predict(X_test)))

# # Chuyển đổi test_scores sang numpy array
# test_scores = np.array(test_scores)

# # Vẽ learning curve
# plt.figure()
# plt.title('Learning Curve (Decision Tree)')
# plt.xlabel('Training examples')
# plt.ylabel('Score')

# # Trực quan hóa khoảng tin cậy của tập huấn luyện và xác thực
# plt.grid()
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
#                  valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

# # Vẽ các đường learning curve
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
# plt.plot(train_sizes, test_scores, 'o-', color="b", label="Test score")

# # Hiển thị legend và biểu đồ
# plt.legend(loc="best")
# plt.show()



# # 5. Vẽ biểu đồ ROC và tính AUC cho phân loại đa lớp

# # Chuyển đổi nhãn sang định dạng one-hot encoding
# from sklearn.preprocessing import label_binarize

# # Lấy danh sách các lớp
# classes = best_model.classes_
# y_valid_bin = label_binarize(y_valid, classes=classes)
# n_classes = y_valid_bin.shape[1]

# # Tính xác suất dự đoán cho mỗi lớp
# y_prob = best_model.predict_proba(X_valid)

# # Tính toán ROC curve và AUC cho mỗi lớp
# fpr = dict()
# tpr = dict()
# roc_auc = dict()

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_valid_bin[:, i], y_prob[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Vẽ ROC cho từng lớp
# plt.figure()

# colors = plt.cm.get_cmap('tab10', n_classes)(range(n_classes))

# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(classes[i], roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Multi-Class')
# plt.legend(loc="lower right")
# plt.show()



# # Print accuracy and classification report on the validation set
# print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred_valid):.2f}")

# # Sử dụng tham số zero_division trong hàm classification_report
# report = classification_report(y_valid, y_pred_valid, zero_division=0)
# print("Classification Report:\n", report)

# # Finally, test the best model on the test set
# y_pred_test = best_model.predict(X_test)
# print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")





# #fit trainnign data into model
# dt_model.fit(X_train, y_train)

# #predict the model on test data
# y_train_pred = dt_model.predict(X_train)
# y_valid_pred = dt_model.predict(X_valid)
# y_test_pred = dt_model.predict(X_test)


# #Model Evaluation
# # 1. Tính toán độ chính xác của mô hình
# # accuracy = accuracy_score(y_valid, y_preds) * 100

# # 2. Lưu ma trận nhầm lẫn
# # cm = confusion_matrix(y_test, y_test_pred)
# # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
# # fig, ax = plt.subplots(figsize=(10, 7))
# # disp.plot(cmap=plt.cm.Blues, ax=ax)
# # plt.ylabel('True label')
# # plt.xlabel('Predicted label')
# # plt.title("Confusion Matrix")

# # In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
# # 3.1. report training set
# report_trainning_set = classification_report(y_train, y_train_pred, target_names=dt_model.classes_, zero_division=0)
# # 3.2. report validation set
# report_validation = classification_report(y_valid, y_valid_pred, target_names=dt_model.classes_, zero_division=0)
# # 3.3. report test set
# report_test_set = classification_report(y_test, y_test_pred, target_names=dt_model.classes_, zero_division=0)

# print(report_validation)
# print(report_trainning_set)
# print(report_test_set)

# # 4. Vẽ sơ đồ learning curve
# train_sizes, train_scores, valid_scores = learning_curve(
#     dt_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')

# # Tính giá trị trung bình và độ lệch chuẩn của các điểm số
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# valid_scores_mean = np.mean(valid_scores, axis=1)
# valid_scores_std = np.std(valid_scores, axis=1)

# # Vẽ biểu đồ learning curve
# # plt.figure(figsize=(10, 6))
# # plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# # plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Validation score")
# # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
# #                  train_scores_mean + train_scores_std, alpha=0.1, color="r")
# # plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
# #                  valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
# # plt.title("Learning Curve")
# # plt.xlabel("Training Examples")
# # plt.ylabel("Score")
# # plt.legend(loc="best")
# # plt.grid(True)

# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.metrics import log_loss
# # from sklearn.model_selection import train_test_split

# # # Lưu trữ log loss
# # train_log_loss = []
# # valid_log_loss = []

# # # Vòng lặp qua các kích thước tập huấn luyện
# # for size in np.linspace(0.1, 0.9, 10):  # Giới hạn trên là 0.9
# #     size = float(size)
# #     print(f"Current training size: {size}")

# #     try:
# #         # Chia tập dữ liệu
# #         X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=size, random_state=42)

# #         # Huấn luyện mô hình
# #         dt_model.fit(X_train_subset, y_train_subset)

# #         # Dự đoán trên tập huấn luyện và tập validation
# #         y_train_probs = dt_model.predict_proba(X_train_subset)
# #         y_valid_probs = dt_model.predict_proba(X_valid)

# #         # Tính toán log loss
# #         train_log_loss.append(log_loss(y_train_subset, y_train_probs))
# #         valid_log_loss.append(log_loss(y_valid, y_valid_probs))
# #     except Exception as e:
# #         print(f"Error while splitting: {e}")
# #         continue

# # # Kiểm tra chiều dài của train_log_loss và valid_log_loss
# # print(f"Training log loss length: {len(train_log_loss)}")
# # print(f"Validation log loss length: {len(valid_log_loss)}")

# # # Vẽ biểu đồ log loss
# # plt.figure(figsize=(10, 6))
# # plt.plot(np.linspace(0.1, 0.9, len(train_log_loss)), train_log_loss, 'o-', color="r", label="Training log loss")
# # plt.plot(np.linspace(0.1, 0.9, len(valid_log_loss)), valid_log_loss, 'o-', color="g", label="Validation log loss")
# # plt.title("Log Loss Curve")
# # plt.xlabel("Training Examples")
# # plt.ylabel("Log Loss")
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import log_loss
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier

# # Dữ liệu và các tham số
# train_sizes = np.linspace(0.1, 0.9, 10)  # Thay đổi kích thước tập huấn luyện
# train_entropy = []
# valid_entropy = []

# for size in train_sizes:
#     X_train_subset, X_valid_subset, y_train_subset, y_valid_subset = train_test_split(X, y, train_size=size, random_state=42)

#     # Huấn luyện mô hình
#     dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42, min_samples_split=10, min_samples_leaf=5)
#     dt_model.fit(X_train_subset, y_train_subset)

#     # Dự đoán trên tập huấn luyện và tập validation
#     y_train_probs = dt_model.predict_proba(X_train_subset)
#     y_valid_probs = dt_model.predict_proba(X_valid_subset)

#     # Tính toán log loss (entropy)
#     train_entropy.append(log_loss(y_train_subset, y_train_probs))
#     valid_entropy.append(log_loss(y_valid_subset, y_valid_probs))

# # Vẽ biểu đồ
# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_entropy, 'o-', color="r", label="Training Entropy")
# plt.plot(train_sizes, valid_entropy, 'o-', color="g", label="Validation Entropy")
# plt.title("Entropy (Log Loss) Curve")
# plt.xlabel("Training Examples")
# plt.ylabel("Entropy (Log Loss)")
# plt.legend(loc="best")
# plt.grid(True)
# plt.show()






# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.metrics import log_loss
# # from sklearn.model_selection import train_test_split
# # from sklearn.tree import DecisionTreeClassifier

# # # Dữ liệu và các tham số
# # train_sizes = np.linspace(0.1, 0.9, 10)  # Thay đổi kích thước tập huấn luyện
# # train_entropy = []
# # valid_entropy = []
# # test_entropy = []

# # for size in train_sizes:
# #     X_train_subset, x_train_temp, y_train_subset, y_train_temp = train_test_split(X, y, train_size=size, random_state=42)
# #     X_valid_temp, X_test_temp, y_valid_temp, y_test_temp = train_test_split(x_train_temp, y_train_temp, test_size=0.5, random_state=42)

# #     # Huấn luyện mô hình
# #     dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
# #     dt_model.fit(X_train_subset, y_train_subset)

# #     # Dự đoán trên tập huấn luyện và tập validation
# #     y_train_probs = dt_model.predict_proba(X_train_subset)
# #     y_valid_probs = dt_model.predict_proba(X_valid_temp)
# #     y_test_probs = dt_model.predict_proba(X_test_temp)

# #     # Tính toán log loss (entropy)
# #     train_entropy.append(log_loss(y_train_subset, y_train_probs))
# #     valid_entropy.append(log_loss(y_valid_temp, y_valid_probs))
# #     test_entropy.append(log_loss(y_test_temp, y_test_probs))

# # # Vẽ biểu đồ
# # plt.figure(figsize=(10, 6))
# # plt.plot(train_sizes, train_entropy, 'o-', color="r", label="Training Entropy")
# # plt.plot(train_sizes, valid_entropy, 'o-', color="g", label="Validation Entropy")
# # plt.plot(train_sizes, test_entropy, 'o-', color="b", label="Test Entropy")
# # plt.title("Entropy (Log Loss) Curve")
# # plt.xlabel("Training Examples")
# # plt.ylabel("Entropy (Log Loss)")
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.show()

# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # from sklearn.metrics import log_loss
# # from sklearn.model_selection import cross_val_score, train_test_split
# # from sklearn.tree import DecisionTreeClassifier

# # # Giả sử bạn đã đọc dữ liệu từ file CSV và đã chia dữ liệu thành X và y
# # data = pd.read_csv("../weather_app/seattle-weather.csv")
# # data = data.dropna()
# # features = ["precipitation", "temp_max", "temp_min", "wind"]
# # X = data[features]
# # y = data["weather"]

# # # Tạo danh sách để lưu log loss cho từng độ sâu
# # max_depths = np.arange(1, 21)  # Thay đổi max_depth từ 1 đến 20
# # valid_entropy = []

# # # Thực hiện Cross-Validation cho mỗi giá trị max_depth
# # for depth in max_depths:
# #     dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)

# #     # Sử dụng Cross-Validation để đánh giá mô hình
# #     scores = cross_val_score(dt_model, X, y, cv=5, scoring='neg_log_loss')  # Đánh giá bằng log loss
# #     valid_entropy.append(-scores.mean())  # Lưu giá trị trung bình log loss (cần đổi dấu)

# # # Vẽ biểu đồ
# # plt.figure(figsize=(10, 6))
# # plt.plot(max_depths, valid_entropy, 'o-', color="g", label="Validation Entropy")
# # plt.title("Entropy (Log Loss) Curve with Cross-Validation")
# # plt.xlabel("Max Depth")
# # plt.ylabel("Entropy (Log Loss)")
# # plt.legend(loc="best")
# # plt.grid(True)
# # plt.show()





# # # Save the model
# # joblib.dump(valueSend, 'decision_tree.pkl')

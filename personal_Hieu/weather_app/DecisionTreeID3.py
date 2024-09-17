#predict weather based on dataset

#import libraries
import base64
import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("./seattle-weather.csv")


#filter data
data = data.dropna()


#features columns to train the model
features = ["precipitation", "temp_max", "temp_min", "wind"]

#Split the datasets into X and y
X = data[features]
y = data["weather"]

#Split the datasets into 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#train set - train model
#validation set - evaluate model
#test set - test model

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)

#fit trainnign data into model
dt_model.fit(X_train, y_train)

#predict the model on test data
y_train__pred = dt_model.predict(X_train)
y_valid_pred = dt_model.predict(X_valid)
y_test_pred = dt_model.predict(X_test)

#Model Evaluation
# 1. Tính toán độ chính xác của mô hình
# accuracy = accuracy_score(y_valid, y_preds) * 100

# 2. Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_model.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")
    #Save the plot to a BytesIO object
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
    #Encode image to base64
plot_url = base64.b64encode(img.getvalue()).decode()

# In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
#3.1. report validation
report_validation = classification_report(y_train, y_train__pred, target_names=dt_model.classes_, zero_division=0)
#3.2. report training set
report_trainning_set = classification_report(y_valid, y_valid_pred, target_names=dt_model.classes_, zero_division=0)
#3.3. report test set
report_test_set = classification_report(y_test, y_test_pred, target_names=dt_model.classes_, zero_division=0)

print(report_validation)
print(report_trainning_set)
print(report_test_set)

# 4. Vẽ sơ đồ learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    dt_model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')

# Tính giá trị trung bình và độ lệch chuẩn của các điểm số
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

# Vẽ biểu đồ learning curve
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
    #Save the plot to a BytesIO object
imgLearningCurve = io.BytesIO()
plt.savefig(imgLearningCurve, format='png')
imgLearningCurve.seek(0)
    #Encode image to base64
learning_curve_url = base64.b64encode(imgLearningCurve.getvalue()).decode()

# # 4. Vẽ sơ đồ entropy
# # Function to calculate entropy at each node
# def calculate_entropy(tree, feature_names):
#     entropy_values = []

#     def entropy(node_id):
#         if node_id == tree.tree_.node_count:
#             return

#         # Compute entropy for the current node
#         if tree.tree_.feature[node_id] != -2:  # -2 indicates leaf nodes
#             feature = feature_names[tree.tree_.feature[node_id]]
#             samples = tree.tree_.weighted_n_node_samples[node_id]
#             impurity = tree.tree_.impurity[node_id]
#             entropy_value = -np.sum((impurity / samples) * np.log2(impurity / samples + 1e-10))  # Added small value to avoid log(0)
#             entropy_values.append(entropy_value)

#             # Traverse to child nodes
#             entropy(tree.tree_.children_left[node_id])
#             entropy(tree.tree_.children_right[node_id])

#     entropy(0)  # Start from the root node
#     return entropy_values

# # Get entropy values
# entropy_values = calculate_entropy(dt_model, features)

# # Plot the entropy values
# plt.figure(figsize=(10, 6))
# plt.plot(entropy_values, marker='o')
# plt.title("Entropy Values Across Tree Nodes")
# plt.xlabel("Node")
# plt.ylabel("Entropy")
# plt.grid(True)
#     #Save the plot to a BytesIO object
# imgEntropy = io.BytesIO()
# plt.savefig(imgEntropy, format='png')
# imgEntropy.seek(0)
#     #Encode image to base64
# entropy_url = base64.b64encode(imgEntropy.getvalue()).decode()

#1. model
#2. report validation
#3. report training set
#4. report test set
#5. ma trận nhầm lẫn (plot_url)
#6. ma trận learning curve


valueSend = {
    'model': dt_model,
    'report_validation': report_validation,
    'report_trainning_set': report_trainning_set,
    'report_test_set': report_test_set,
    'plot_url': plot_url,
    'learning_curve_url': learning_curve_url
}

# Save the model
joblib.dump(valueSend, 'decision_tree.pkl')

#import libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("../../seattle-weather.csv", index_col=0)

#features columns to train the model
features = ["precipitation", "temp_max", "temp_min", "wind"]

#Split the datasets into X and y
X = data[features]
y = data["weather"]

#Split the datasets into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

#fit trainnign data into model
dt_model.fit(X_train, y_train)

# Save the model (bỏ qua nếu không cần thiết)
# joblib.dump(dt_model, 'weather_model.pkl')

#predict the model on test data
y_preds = dt_model.predict(X_valid)

#predict with new input
new_input = pd.DataFrame({
    'precipitation': [4.3],
    'temp_max': [13.9],
    'temp_min': [10],
    'wind': [2.8]
})

print(dt_model.predict(new_input))

# Pain the model graphics
# compare the prediction with the actual values
cm = confusion_matrix(y_valid, y_preds)

print(cm)

# Function to calculate loss (log loss)
def calculate_loss(y_true, y_pred_proba):
    """
    Hàm tính cross-entropy loss (log loss)

    Parameters:
    y_true: nhãn thực tế
    y_pred_proba: xác suất dự đoán từ mô hình

    Returns:
    log loss: giá trị lỗi của mô hình
    """
    return log_loss(y_true, y_pred_proba)

# Predict probabilities for validation data
y_pred_proba = dt_model.predict_proba(X_valid)

# Tính toán loss
loss = calculate_loss(y_valid, y_pred_proba)
print(f"Training Loss (Log Loss): {loss:.4f}")

#test entropy
def entropy(column):
    probabilities = column.value_counts(normalize=True)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

print(f"entropy: {entropy(data['weather'])}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# Hàm tính entropy từ các giá trị tại mỗi node
def calculate_node_entropy(tree, node_id):
    # Lấy số lượng mẫu tại node hiện tại
    node_samples = tree.tree_.value[node_id][0]

    # Tính xác suất của mỗi lớp tại node này
    probabilities = node_samples / np.sum(node_samples)

    # Tính entropy
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

# Hàm tính entropy cho toàn bộ cây
def calculate_tree_entropy(tree):
    node_count = tree.tree_.node_count
    entropy_values = []

    # Duyệt qua tất cả các node
    for node_id in range(node_count):
        if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:  # Node lá
            entropy_values.append(calculate_node_entropy(tree, node_id))
        else:  # Node nội bộ
            entropy_values.append(calculate_node_entropy(tree, node_id))

    return entropy_values


# Tính entropy cho toàn bộ cây
entropy_values = calculate_tree_entropy(dt_model)

# Vẽ biểu đồ entropy
plt.figure(figsize=(10, 6))
plt.plot(entropy_values, marker='o')
plt.title("Entropy Values Across Tree Nodes")
plt.xlabel("Node")
plt.ylabel("Entropy")
plt.grid(True)
plt.show()


# #Model Evaluation
# # 1. Tính toán độ chính xác của mô hình
# accuracy = accuracy_score(y_valid, y_preds) * 100
# print(f"Accuracy: {accuracy:.2f}%")

# # 2. In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
# report = classification_report(y_valid, y_preds, target_names=dt_model.classes_)
# print("Classification Report:\n", report)

# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# # Vẽ sơ đồ cây quyết định
# plt.figure(figsize=(15,10))
# plot_tree(dt_model, feature_names=features, class_names=dt_model.classes_, filled=True, rounded=True)
# plt.title("Decision Tree Visualization")
# plt.show()



from sklearn.model_selection import learning_curve

# Vẽ learning curve
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
plt.show()


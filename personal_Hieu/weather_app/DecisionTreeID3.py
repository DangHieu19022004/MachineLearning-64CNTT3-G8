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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#get the dataset
data = pd.read_csv("../../seattle-weather.csv")


#filter data
data = data.dropna()


#features columns to train the model
features = ["precipitation", "temp_max", "temp_min", "wind"]

#Split the datasets into X and y
X = data[features]
y = data["weather"]

#Split the datasets into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)

#fit trainnign data into model
dt_model.fit(X_train, y_train)

#predict the model on test data
y_preds = dt_model.predict(X_valid)

#Model Evaluation
# 1. Tính toán độ chính xác của mô hình
accuracy = accuracy_score(y_valid, y_preds) * 100

# 2. Lưu ma trận nhầm lẫn
cm = confusion_matrix(y_valid, y_preds)
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

# 3. In báo cáo phân loại (precision, recall, F1-score cho từng lớp)
report = classification_report(y_valid, y_preds, target_names=dt_model.classes_)

# 4. Vẽ sơ đồ entropy
# Function to calculate entropy at each node
def calculate_entropy(tree, feature_names):
    entropy_values = []

    def entropy(node_id):
        if node_id == tree.tree_.node_count:
            return

        # Compute entropy for the current node
        if tree.tree_.feature[node_id] != -2:  # -2 indicates leaf nodes
            feature = feature_names[tree.tree_.feature[node_id]]
            samples = tree.tree_.weighted_n_node_samples[node_id]
            impurity = tree.tree_.impurity[node_id]
            entropy_value = -np.sum((impurity / samples) * np.log2(impurity / samples + 1e-10))  # Added small value to avoid log(0)
            entropy_values.append(entropy_value)

            # Traverse to child nodes
            entropy(tree.tree_.children_left[node_id])
            entropy(tree.tree_.children_right[node_id])

    entropy(0)  # Start from the root node
    return entropy_values

# Get entropy values
entropy_values = calculate_entropy(dt_model, features)

# Plot the entropy values
plt.figure(figsize=(10, 6))
plt.plot(entropy_values, marker='o')
plt.title("Entropy Values Across Tree Nodes")
plt.xlabel("Node")
plt.ylabel("Entropy")
plt.grid(True)
    #Save the plot to a BytesIO object
imgEntropy = io.BytesIO()
plt.savefig(imgEntropy, format='png')
imgEntropy.seek(0)
    #Encode image to base64
entropy_url = base64.b64encode(imgEntropy.getvalue()).decode()


valueSend = {
    'model': dt_model,
    'accuracy': accuracy,
    'report': report,
    'plot_url': plot_url,
    'entropy_url': entropy_url
}

# Save the model
joblib.dump(valueSend, 'decision_tree.pkl')

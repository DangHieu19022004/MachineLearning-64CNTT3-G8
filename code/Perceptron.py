import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('./seattle-weather.csv')

# Filter out missing data
data = data.dropna()

# Features to use for training
features = ["precipitation", "temp_max", "temp_min", "wind"]

# Split data into X and y
X = data[features]
y = data["weather"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data: 70% training, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train Perceptron model
perceptron_model = Perceptron(max_iter=100, random_state=42, eta0=0.01, tol=0.001)
perceptron_model.fit(X_train, y_train)

# Predict on the test set
y_test_pred = perceptron_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.show()

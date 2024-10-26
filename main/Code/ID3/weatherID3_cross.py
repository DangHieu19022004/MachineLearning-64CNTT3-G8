#predict weather based on dataset

#import libraries
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


#filter data
data = data.dropna()
data.drop(['date'], axis=1, inplace=True)

#Split the datasets into X and y
X = data[["precipitation", "temp_max", "temp_min", "wind"]]
y = data["weather"]

#Split the datasets into 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# # Define the parameter grid for cross-validation
param_grid = {
    'max_depth': [1, 2, 3, 5, 7, 9, None],
    'min_samples_split': [2, 3, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'splitter': ['best', 'random'],
    'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

#trainning the model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# # Define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)

# Train with cross-validation
grid_search.fit(X_train, y_train)

# Output the best parameters found
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

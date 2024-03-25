# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# k-Nearest Neighbor

# %% load packages
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# %% load the data
dataset = load_breast_cancer()

# %% prepare the data
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = X[['mean smoothness', 'mean concavity', 'radius error']]
y = pd.Categorical.from_codes(dataset.target, dataset.target_names)
y = pd.get_dummies(y, drop_first=True)

print(X)
print(y)

# %% split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %% specify and fit the kNN model
knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan')
knn.fit(X_train, y_train.values.ravel())

# %% use the model to predict values
y_pred = knn.predict(X_test)

# %% print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# %% print accuracy
accuracy_score(y_test, y_pred)
# console output: 0.8671328671328671


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert y_test to the same format as y_pred
y_test_encoded = np.argmax(y_test.values, axis=1)

# If y_pred is 2D (multiple classes), no need to adjust
if y_pred.ndim == 2:
    y_pred_encoded = np.argmax(y_pred, axis=1)
# If y_pred is 1D (binary classification), convert it to 2D
else:
    y_pred_encoded = y_pred

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_encoded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=dataset.target_names, yticklabels=dataset.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("test_code_from_uny_confusion_matrix.png")

# Accuracy Score
plt.figure(figsize=(6, 4))
sns.barplot(x=['Accuracy'], y=[accuracy_score(y_test_encoded, y_pred_encoded)])
plt.ylim(0, 1)
plt.title('Accuracy Score')
plt.savefig("test_code_from_uny_accuracy_score.png")



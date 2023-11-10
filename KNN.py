import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv('/Users/jahnavivemuri/Downloads/train.csv')

# Assuming your target variable is 'Survived'
X = data.drop('Survived', axis=1)
y = data['Survived']

# Convert non-numeric columns to numeric type
X_numeric = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values, you can use different strategies based on your data
X_numeric.fillna(0, inplace=True)

# Scale the data using standard scaler
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled

X_scaled = standard_scaler(X_numeric)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Determine the K value and create a visualization of the accuracy
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot the accuracy for different K values
plt.plot(k_values, accuracies)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Classifier Accuracy for Different K Values')
plt.show()

# Report the best K value
best_k = k_values[np.argmax(accuracies)]
print(f"Best K value: {best_k}")

# Run 5-fold cross validations and report mean and standard deviation
cross_val_scores = cross_val_score(knn, X_scaled, y, cv=5)
print(f"Mean Cross Validation Score: {np.mean(cross_val_scores)}")
print(f"Standard Deviation of Cross Validation Score: {np.std(cross_val_scores)}")

# Evaluate using confusion matrix
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print(conf_matrix)

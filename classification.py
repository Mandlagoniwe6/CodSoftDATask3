import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

iris_data = pd.read_csv("IRIS.csv", header=0, sep=",")

# Pairplot to visualize relationships between features
sns.pairplot(iris_data, hue="species", markers=["o", "s", "D"])
plt.show()

# Correlation Heatmap
numeric_features = iris_data.drop("species", axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode the species column which is the target variable
le = LabelEncoder()
iris_data['species'] = le.fit_transform(iris_data['species'])

# Separate features and target variable
X = iris_data.drop("species", axis=1)
y = iris_data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf_classifier = RandomForestClassifier(random_state=42)

# Perform a cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Train the model on the full training set
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Dimensionality Reduction 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data in reduced dimensions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris_data['species'], palette='viridis', legend='full')
plt.title("PCA Projection of Iris Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Prediction on New Data
new_data = np.array([[5.0, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.0, 6.0, 1.8]])  # Setosa, Versicolor, Virginica
predictions = rf_classifier.predict(new_data)

# Decode predictions back to species names
decoded_predictions = le.inverse_transform(predictions)
print("Predictions for new data:")
print(decoded_predictions)

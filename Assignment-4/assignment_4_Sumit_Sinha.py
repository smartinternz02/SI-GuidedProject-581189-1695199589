import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# 1.Load the Dataset
dataset = pd.read_csv("C:/Users/Sumit/AI ML SmartBridge/Assignment-4/winequality-red.csv")
print(dataset.head())

# 2.Data preprocessing including visualization
plt.figure(figsize=(12, 6))
sns.histplot(dataset['quality'], color='red', bins=5, kde=True)
plt.xlabel('Wine Quality')
plt.ylabel('Frequency')
plt.title('Red Wine Quality Distribution')
plt.show()

# 3.Machine Learning Model building
X = dataset.drop('quality', axis=1)
y = dataset['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# 4.Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {f1:.2f}")
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 5.Test with random observation
new_observation = pd.DataFrame({
    'fixed acidity': [7.0],
    'volatile acidity': [0.3],
    'citric acid': [0.2],
    'residual sugar': [2.0],
    'chlorides': [0.08],
    'free sulfur dioxide': [15],
    'total sulfur dioxide': [50],
    'density': [0.995],
    'pH': [3.3],
    'sulphates': [0.6],
    'alcohol': [10.5]
})
predicted_quality = rf_classifier.predict(new_observation)
print(f"Predicted Wine Quality: {predicted_quality[0]}")


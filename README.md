# Import libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Titanic dataset using Seaborn
titanic_data = sns.load_dataset('titanic')

# Display the first few rows of the dataset
titanic_data.head()

# Data preprocessing
titanic_data.drop(['embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], axis=1, inplace=True)
titanic_data.dropna(inplace=True)
titanic_data['sex'] = titanic_data['sex'].map({'male': 0, 'female': 1})

# Split the data into features (X) and target variable (y)
X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the accuracy and confusion matrix
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Display the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

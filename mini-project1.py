import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import category_encoders as ce # You may need to: pip install category_encoders

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=col_names)

# Preview data
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Target variable distribution
print(df['class'].value_counts())

# Split into features and target
X = df.drop(['class'], axis=1)
y = df['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train.head())

# Initialize the model
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# Fit the model
clf_gini.fit(X_train, y_train)

# Predict results
y_pred_gini = clf_gini.predict(X_test)

# Check accuracy
print(f"Model accuracy score with Gini Index: {accuracy_score(y_test, y_pred_gini):.4f}")


# Initialize model with Entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# Fit model
clf_en.fit(X_train, y_train)

# Predict
y_pred_en = clf_en.predict(X_test)

print(f"Model accuracy score with Entropy: {accuracy_score(y_test, y_pred_en):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_en)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred_en))


from sklearn import tree

plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train), 
               feature_names=X_train.columns, 
               class_names=y_train.unique(), 
               filled=True)
plt.show()
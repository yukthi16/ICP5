#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the glass dataset
glass = pd.read_csv("glass.csv")

# Split the dataset into features and target
X = glass.iloc[:, :-1]
y = glass.iloc[:, -1]

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

# Fit the Naive Bayes model on the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict the target on the test data
y_pred = gnb.predict(X_test)

# Evaluate the model on the test data
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

# Load the glass dataset
glass = pd.read_csv("glass.csv")

# Split the dataset into features and target
X = glass.iloc[:, :-1]
y = glass.iloc[:, -1]

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# Fit the linear SVM model on the training data
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Predict the target on the test data
y_pred = svm_linear.predict(X_test)

# Evaluate the model on the test data
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





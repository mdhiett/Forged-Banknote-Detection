#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = pd.read_csv("C:\\banknote.csv")


# In[3]:


data.head()


# In[4]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


X = data[['V1', 'V2', 'V3', 'V4']]


y = data['Class']


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


rf_model.fit(X, y)


# In[6]:


feature_importances = rf_model.feature_importances_


# In[7]:


X = data[['V1', 'V2', 'V3', 'V4']]


feature_names = X.columns


feature_importances = rf_model.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()


# In[9]:


feature_importances = rf_model.feature_importances_


feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)


print("\nRandom Forest Model Summary:")
print(rf_model)


# In[10]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
y_pred = rf_model.predict(X)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Class 1', 'Class 2'],
            yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[12]:


X = data.drop('Class', axis=1)  
y = data['Class']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


model = LogisticRegression()


# In[15]:


model.fit(X_train, y_train)


# In[16]:


predictions = model.predict(X_test)


# In[17]:


accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, predictions))


# In[ ]:





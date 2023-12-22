#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import numpy as np


# In[3]:


data = pd.read_csv("C:\\banknote.csv")


# In[4]:


data.head()


# In[5]:


data.info


# In[6]:


from sklearn.ensemble import RandomForestClassifier

# Features
X = data[['V1', 'V2', 'V3', 'V4']]

# Target variable
y = data['Class']

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with features and target variable
rf_model.fit(X, y)



# In[7]:


data.info


# In[8]:


feature_importances = rf_model.feature_importances_


# In[9]:


import matplotlib.pyplot as plt
import pandas as pd




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


# In[10]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


y_pred = rf_model.predict(X)


conf_matrix = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Class 1', 'Class 2'],
            yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[11]:


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


# In[ ]:





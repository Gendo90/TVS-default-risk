#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Update sklearn to prevent version mismatches
get_ipython().system('pip install sklearn --upgrade')


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


# # Import Data

# In[2]:


#Import cleaned csv file
data = pd.read_csv("output.csv")
data.columns


# In[3]:


#Select only relevant columns
model_data = data[[
    'EMI', 
    'Loan Amount', 
    'Maximum amount sanctioned for any Two wheeler loan',
    'Age at which customer has taken the loan', 
    'Rate of Interest', 
    'Number of times 30 days past due in last 6 months', 
    'Maximum MOB (Month of business with TVS Credit)', 
    'Number of times 60 days past due in last 6 months', 
    'Number of loans', 
    'Maximum amount sanctioned in the Live loans', 
    'Number of times 90 days past due in last 3 months', 
    'Tenure', 
    'Number of times bounced while repaying the loan',
    'Target variable ( 1: Defaulters / 0: Non-Defaulters)'
]]


# In[4]:


#Rename Dependent Variable Column to "Class"
model_data = model_data.rename(columns={"Target variable ( 1: Defaulters / 0: Non-Defaulters)": "Class"})
model_data.head()


# # Train the Model

# In[5]:


# Assign X (data) and y (target)
X = model_data.drop('Class', axis=1)
y = model_data['Class']
print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")


# In[6]:


#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=5)


# #Resample Data using Oversampling Technique

# In[7]:


# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_defaulted = X[X.Class==0]
defaulted = X[X.Class==1]

print(not_defaulted.Class.value_counts())
print(defaulted.Class.value_counts())


# In[8]:


from sklearn.utils import resample
# upsample minority
defaulted_upsampled = resample(defaulted,
                               replace=True,
                               n_samples=len(not_defaulted),
                               random_state=27)
# combine majority and upsampled minority
upsampled = pd.concat([not_defaulted, defaulted_upsampled])
upsampled.Class.value_counts()


# In[9]:


#redefine the X and y training samples
y_train = upsampled.Class
X_train = upsampled.drop('Class', axis=1)


# In[10]:


#Define the classifier as a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier

#fit the training samples in the classifier
classifier.fit(X_train, y_train)
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


# # Assessment 

# In[11]:


#Compare with test data
predictions = classifier.predict(X_test)

print(sum(predictions), sum(y_test))

total = 0
default_num = 0

for i in range(len(predictions)):
    if(predictions[i] == y_test.array[i]):
        total +=1
        if(predictions[i]==1):
            default_num+=1
print("-------" * 2)            
print(default_num, sum(y_test))
print("-------" * 2)
print(total/len(predictions))


# # Grid Search

# In[12]:


#Import Grid Search
from sklearn.model_selection import GridSearchCV

#Define param_grid
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

#define clf
clf = GridSearchCV(classifier, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)


# In[ ]:


best_clf = clf.fit(X_train, y_train)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[6]:



pip install xgboost


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[9]:


df=pd.read_csv("C:/Users/21294596/Downloads/archive/WineQT.csv")
print(df.head())


# In[10]:


df.info()


# In[12]:


df.describe().T


# In[13]:


df.isnull().sum()


# In[16]:


for col in df.columns:
    if df[col].isnull().sum()>0:
        df[col]=df[col].fillna(df[col].mean())
        
df.isnull().sum().sum()


# In[24]:


df.hist(bins=20,figsize=(10, 10))
plt.show()


# In[25]:


plt.bar(df["quality"],df["alcohol"])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[28]:


plt.figure(figsize=(12,12))
sb.heatmap(df.corr()>0.7, annot=True,cbar=False)
plt.show()


# In[30]:


df=df.drop('total sulfur dioxide', axis=1)


# In[31]:


df['best quality']= [1 if x > 5 else 0 for x in df.quality]


# In[32]:


df.replace({'white':1, 'red':0},inplace=True)


# In[33]:


features=df.drop(["quality","best quality"], axis=1)
target=df['best quality']

xtrain,xtest,ytrain,ytest=train_test_split(features, target,test_size=0.2, random_state=40)
xtrain.shape,xtrain.shape


# In[34]:


norm=MinMaxScaler()
xtrain=norm.fit_transform(xtrain)
xtest=norm.transform(xtest)


# In[42]:


models=[LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain,ytrain)
    print(f'{models[i]}: ')
    print("Traning accuracy:", metrics.roc_auc_score(ytrain,models[i].predict(xtrain)))
    print("Validation accuracy:", metrics.roc_auc_score(ytest,models[i].predict(xtest)))
    print()


# In[44]:


metrics.plot_confusion_matrix(models[i],xtest,ytest)
plt.show()


# In[ ]:





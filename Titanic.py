#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os


# In[3]:


titanic=pd.read_csv("/Users/afaq/Downloads/train.csv")


# In[4]:


titanic


# In[5]:


titanic.describe()


# In[6]:


titanic.info()


# In[7]:


titanic.isnull().sum()


# In[8]:


for i in titanic.describe().columns:
    sns.boxplot(titanic[i])
    plt.show()


# In[9]:


titanic.Cabin.value_counts()


# In[10]:


titanic.Embarked.value_counts()


# In[11]:


titanic["Age"].fillna(titanic["Age"].mean(),inplace=True)


# In[12]:


titanic['Embarked'].fillna("S",inplace= True)


# In[13]:


T1=titanic.drop(columns='Cabin',axis=1)


# In[15]:


T1.shape


# In[18]:


T1.isnull().sum()


# In[19]:


sns.boxplot(np.sqrt(T1['Fare']))


# In[20]:


sns.boxplot(np.log(T1['Fare']))


# In[22]:


Fare_=np.where(T1['Fare']<0,0,np.sqrt(T1['Fare']))


# In[23]:


T1['Faresqrt']=Fare_


# In[24]:


T1=T1.drop(columns='Fare',axis=1)


# In[25]:


T1


# In[27]:


for i in T1.describe().columns:
    sns.distplot(T1[i])
    plt.show()


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[32]:


dummy=pd.get_dummies(T1)


# In[33]:


T1


# In[34]:


dummy


# In[35]:


X = T1.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = T1['Survived']


# In[36]:


X.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[37]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=10)


# In[38]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[39]:


model = LogisticRegression()


# In[40]:


model.fit(X_train, Y_train)


# In[41]:


X_train_prediction = model.predict(X_train)


# In[42]:


from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[43]:


from sklearn import tree


# In[44]:


dt=tree.DecisionTreeClassifier()


# In[45]:


dt.fit(X_train,Y_train)


# In[46]:


Y_Pred=dt.predict(X_train)


# In[47]:


from sklearn import metrics


# In[48]:


print("Accuracy - ",metrics.accuracy_score(Y_train,Y_Pred))


# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


rf=RandomForestClassifier(n_estimators=100,bootstrap=False)


# In[51]:


rf.fit(X_train,Y_train)


# In[52]:


Y_Pred=rf.predict(X_train)


# In[53]:


print("Accuracy = ",metrics.accuracy_score(Y_train,Y_Pred))


# In[ ]:





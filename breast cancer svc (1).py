#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn 


# In[4]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()


# In[5]:


cancer.feature_names


# In[6]:


cancer.target_names


# In[8]:


df=pd.DataFrame(cancer.data,columns=cancer.feature_names)


# In[9]:


df.head()


# In[10]:


df['target']=cancer.target
df.head()


# In[11]:


df[df.target==1].head()


# In[14]:


df['cancer class']=df.target.apply(lambda x:cancer.target_names[x])


# In[15]:


df0=df[df.target==0].head(50)
df1=df[df.target==1].head(50)


# In[17]:


import matplotlib.pyplot as plt
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.scatter(df0['mean radius'],df0['mean texture'],color='red',marker='*')
plt.scatter(df1['mean radius'],df1['mean texture'],color='green',marker='+')


# In[18]:


from sklearn .model_selection import train_test_split
x=df.drop(['target','cancer class'],axis='columns')
y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=4)


# In[19]:


from sklearn.svm import SVC
mymodel=SVC()


# In[20]:


mymodel.fit(x_train,y_train)


# In[24]:


pred=mymodel.predict(x_test)


# In[23]:


mymodel.score(x_test,y_test)


# In[26]:


from sklearn.metrics import confusion_matrix
cm=np.array(confusion_matrix(y_test,pred))
cm


# In[27]:


squares=[]
for i in range(10):
    squares.append(i*i)
squares    


# In[29]:


squares=[i*i for i in range(10)]  #expression for x in list iterable
squares


# In[30]:


m=[[j for j in range(5)] for i in range(5)]
m


# In[33]:


l=[1,2,3,4]
new_l=[i**3if i%2==0  else i**2 for i in l]
new_l


# In[ ]:





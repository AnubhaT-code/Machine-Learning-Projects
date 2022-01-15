#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/AnubhaT-code/Machine-Learning-Projects/main/wisc_bc_data.csv")


# In[3]:


data.head(10)


# In[4]:


# Count of number of rows and columns
data.shape


# In[5]:


# Count of null values in each column
data.isnull().sum()


# In[6]:


# Datatypes
data.dtypes


# In[7]:


# Count of Malignant or Benign tumours
data['diagnosis'].value_counts()


# In[8]:


# Visualization
sns.countplot(data['diagnosis'], label='count')


# In[9]:


data['diagnosis'] = data['diagnosis'].replace({'B':1,'M':0})


# In[10]:


data.sample(10)


# In[11]:


sns.pairplot(data.iloc[:,1:8], hue='diagnosis')


# In[12]:


# one hot encoding
# data = pd.get_dummies(data, columns=['diagnosis'])
data.sample(10)


# In[13]:


data.dtypes


# In[14]:


# create a pairplot
sns.pairplot(data.iloc[:,1:8])


# In[15]:


data.head()


# In[16]:


# Get the correlation of the columns
data.iloc[:,1:12].corr()


# In[17]:


# Visualization of correlation
plt.figure(figsize=(10,10))
sns.heatmap(data.iloc[:,1:12].corr(), annot=True, fmt='.0%')


# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[19]:


# Split the data set into independent (X) and dependent (Y) data sets
X = data.iloc[:,2:31].values
Y = data.iloc[:,1].values


# In[20]:


# Splitting data set into 75% training and 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25 , random_state = 0)


# In[21]:


# Scale the data (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[22]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[23]:


X = data.drop("diagnosis",axis=1)
y = data["diagnosis"]


# In[24]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# In[25]:


predictions= model.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)


# In[27]:


from sklearn import metrics 
print(metrics.classification_report(Y_test, predictions))


# In[28]:


dtree = DecisionTreeClassifier(criterion='entropy',random_state=1)
dtree.fit(X_train,Y_train)


# In[29]:


print(dtree.score(X_train,Y_train))
print(dtree.score(X_test,Y_test))


# In[30]:


dtree = DecisionTreeClassifier(criterion='gini',max_depth = 5,random_state=1)
dtree.fit(X_train,Y_train)
print(dtree.score(X_train,Y_train))
print(dtree.score(X_test,Y_test))


# In[31]:


from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier( n_estimators=50,base_estimator=dtree,random_state=1)
bgcl = bgcl.fit(X_train,Y_train)
y_predict = bgcl.predict(X_test)
print(bgcl.score(X_train,Y_train))
print(bgcl.score(X_test,Y_test))


# In[32]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier( n_estimators=500,random_state=1)
abcl = abcl.fit(X_train,Y_train)
y_predict = abcl.predict(X_test)
print(abcl.score(X_train,Y_train))
print(abcl.score(X_test,Y_test))


# In[33]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = BaggingClassifier( n_estimators=1000,random_state=1)
gbcl = gbcl.fit(X_train,Y_train)
y_predict = gbcl.predict(X_test)
print(gbcl.score(X_train,Y_train))
print(gbcl.score(X_test,Y_test))


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier( n_estimators=50,random_state=1,max_features=7)
rfcl = rfcl.fit(X_train,Y_train)
y_predict = rfcl.predict(X_test)
print(rfcl.score(X_train,Y_train))
print(rfcl.score(X_test,Y_test))


# In[35]:


# test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,predictions)
TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)
print("Testing Accuracy = ", (TP + TN) / (TP + TN + FN + FP))


# In[36]:


cm = metrics.confusion_matrix(Y_test, predictions, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                    columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





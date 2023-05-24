#!/usr/bin/env python
# coding: utf-8

# # California Housing Price Prediction

# ## Background of Problem Statement

# Biro Sensus AS telah menerbitkan Data Sensus California yang memiliki 10 jenis metrik seperti populasi, pendapatan rata-rata, harga perumahan rata-rata, dan sebagainya untuk setiap kelompok blok di California. Dataset juga berfungsi sebagai masukan untuk pelingkupan proyek dan mencoba menentukan persyaratan fungsional dan nonfungsional untuknya.

# ## Problem Objective 

# Proyek ini bertujuan membangun model harga rumah untuk memprediksi nilai rata-rata rumah di California menggunakan kumpulan data yang disediakan. Model ini harus belajar dari data dan dapat memprediksi harga rumah rata-rata di kabupaten mana pun, mengingat semua metrik lainnya.
# 
# Distrik atau kelompok blok adalah unit geografis terkecil yang Biro Sensus AS menerbitkan data sampel (kelompok blok biasanya memiliki populasi 600 hingga 3.000 orang). Ada 20.640 kabupaten dalam kumpulan data proyek.

# ## Data Understanding

# Domain: Keuangan dan Perumahan
# 
# Analisis Tugas yang harus dilakukan:
# 
# 1. Bangun model harga rumah untuk memprediksi nilai rata-rata rumah di California menggunakan kumpulan data yang disediakan.
# 
# 2. Latih model untuk belajar dari data untuk memprediksi harga rumah rata-rata di kabupaten mana pun, dengan mempertimbangkan semua metrik lainnya.
# 
# 3. Prediksikan harga rumah berdasarkan median_income dan plot grafik regresi untuk itu.

# #### Import Library

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report


# In[2]:


# load the data
housingData = pd.read_excel('1553768847_housing.xlsx')
housingData.head()


# In[3]:


housingData.info()


# In[4]:


housingData.describe()


# In[5]:


housingData.shape


# In[6]:


plt.figure(figsize=(15,10))
sns.heatmap(housingData.corr(),annot=True)


# In[7]:


sns.pairplot(housingData, diag_kind = 'kde')


# In[8]:


housingData.hist(bins=50, figsize=(20,15))
plt.show()


# In[9]:


X = housingData.iloc[:, :-1].values
y = housingData.iloc[:, [-1]].values


# In[10]:


# Handle missing values 
from sklearn.impute import SimpleImputer

missingValueImputer = SimpleImputer()
X[:, :-1] = missingValueImputer.fit_transform(X[:, :-1])
y = missingValueImputer.fit_transform(y)


# ## Data Preparation

# In[11]:


from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, -1] = X_labelencoder.fit_transform(X[:, -1])


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)


# ## Model Development

# In[14]:


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)


# In[15]:


predictionLinear = linearRegression.predict(X_test)


# In[16]:


from sklearn.metrics import mean_squared_error
mseLinear = mean_squared_error(y_test, predictionLinear)
print('Root mean squared error (RMSE) from Linear Regression = ')
print(mseLinear)


# In[17]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, y_train)


# In[18]:


predictionDT = DTregressor.predict(X_test)


# In[19]:


from sklearn.metrics import mean_squared_error
mseDT = mean_squared_error(y_test, predictionDT)
print('Root mean squared error from Decision Tree Regression = ')
print(mseDT)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, y_train)


# In[21]:


predictionRF = RFregressor.predict(X_test)


# In[22]:


from sklearn.metrics import mean_squared_error
mseRF = mean_squared_error(y_test, predictionRF)
print('Root mean squared error from Random Forest Regression = ')
print(mseRF)


# In[23]:


X_train_median_income = X_train[: , [7]]
X_test_median_income = X_test[: , [7]]


# In[24]:


from sklearn.linear_model import LinearRegression
linearRegression2 = LinearRegression()
linearRegression2.fit(X_train_median_income, y_train)


# In[25]:


predictionLinear2 = linearRegression2.predict(X_test_median_income)


# In[26]:


plt.scatter(X_train_median_income, y_train, color = 'green')
plt.plot (X_train_median_income, 
          linearRegression2.predict(X_train_median_income), color = 'red')
plt.title ('compare Training result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()


# In[27]:


plt.scatter(X_test_median_income, y_test, color = 'blue')
plt.plot (X_train_median_income, 
          linearRegression2.predict(X_train_median_income), color = 'red')
plt.title ('compare Testing result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()


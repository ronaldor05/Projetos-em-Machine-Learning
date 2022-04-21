#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('c:/dados/bigmart.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.drop(labels='Item_Identifier', axis=1, inplace=True)


# In[ ]:





# In[ ]:


#verificando dados faltantes e tipo de coluna
df.isna().sum()


# In[ ]:


#Tipos de dados de cada atributo (variáveis)
df.info()


# In[ ]:


plt.scatter(df['Outlet_Establishment_Year'], df['Item_Outlet_Sales']);


# In[ ]:


df['Outlet_Establishment_Year'] = df['Outlet_Establishment_Year'].astype("category")


# In[ ]:


#visualizando informações sobre dados
categories = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Establishment_Year',
              'Outlet_Location_Type', 'Outlet_Type']
values = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
target = 'Item_Outlet_Sales'
for column in categories:
    print(df[column].value_counts(), end='\n\n')


# In[ ]:


df[values].hist(figsize=(10,8));


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df[target].hist(figsize=(10,8));


# In[ ]:


#limpando os dados
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
df['Outlet_Size'].fillna("Unknown", inplace=True)


# In[ ]:


low_fat = lambda x: ((x == 'LF' or x == 'low fat') and 'Low Fat') or x
reg_fat = lambda x: (x == 'reg' and 'Regular') or x


# In[ ]:


df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(low_fat)
df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(reg_fat)


# In[ ]:


df['Item_Fat_Content'].value_counts()


# In[ ]:


df.isna().sum()


# In[ ]:


from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
df[values] = scaler.fit_transform(df[values])


# In[ ]:


df[values].hist(figsize=(10,8));


# In[ ]:


#tratando os dados categoricos
df = pd.get_dummies(df, drop_first=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


#separando dados de treino e teste
X = df.drop(labels=target, axis=1)
y = df[target]


# In[ ]:


y.describe()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)


# In[ ]:


#treinando o modelo de regressão linear
from sklearn.linear_model import LinearRegression


# In[ ]:


clf = LinearRegression()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)


# In[ ]:


#medindo o desempenho do modelo
from sklearn.metrics import mean_squared_error


# In[ ]:


print("RMSE treino:", np.sqrt(mean_squared_error(y_train, clf.predict(X_train))))
print("RMSE teste:", np.sqrt(mean_squared_error(y_test, y_predict)))


# In[ ]:


#coeficiente de hipótese
clf.intercept_, clf.coef_


# In[ ]:





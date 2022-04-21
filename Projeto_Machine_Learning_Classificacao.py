#!/usr/bin/env python
# coding: utf-8

# In[41]:



from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#Features (1 sim 0 não)
#Pelo longo
#perna curta?
#Faz auau?
porco1 = [0,1,0]
porco2 = [0,1,1]
porco3 = [1,1,0]

cachorro1 =[0,1,1]
cachorro2 =[1,0,1]
cachorro3 =[1,1,1]


# In[8]:


dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]


# In[12]:


classes = [1,1,1,0,0,0]


# In[13]:


model = LinearSVC()


# In[14]:


model.fit(dados, classes)


# In[15]:


animal_misterioso = [1,1,1]


# In[16]:


model.predict([animal_misterioso])


# In[19]:


misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1, misterio2, misterio3]
model.predict(testes)


# In[20]:


previsoes = model.predict(testes)


# In[21]:


testes_classes =[0,1,1]


# In[22]:


previsoes == testes_classes


# In[23]:


corretos = (previsoes == testes_classes).sum()


# In[24]:


total = len(testes)


# In[26]:


taxa_de_acerto = corretos/total


# In[28]:


print("taxa de acerto:", taxa_de_acerto*100)


# In[29]:


from sklearn.metrics import accuracy_score


# In[47]:


taxa_de_acerto = accuracy_score


# In[31]:


(testes_classes, previsoes)


# In[37]:


print("taxa de acerto", taxa_de_acerto)


# In[48]:


from sklearn.metrics import accuracy_score
taxa_de_acerto = accuracy_score
(testes_classes, previsoes)
print("Taxa de Acerto", taxa_de_acerto)


# In[44]:


taxa_de_acerto = accuracy_score
(testes_classes, previsoes)


# In[42]:


print("Taxa de Acerto", taxa_de_acerto)


# In[49]:


from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(testes_classes, previsoes)
print("Taxa de acerto", taxa_de_acerto * 100)


# In[ ]:





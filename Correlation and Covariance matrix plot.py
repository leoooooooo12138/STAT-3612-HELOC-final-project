#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv("https://raw.githubusercontent.com/ajzhanghku/Stat3612/master/HelocData.csv")
print(df)


# In[4]:


import seaborn as sns
corr=df.corr()
ax=sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20,220, n=20),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


# In[5]:


import matplotlib.pyplot as plt
plt.matshow(df.corr())
plt.show()


# In[9]:


rs=np.random.RandomState(0)
df=pd.DataFrame(rs.rand(23,23))
corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[15]:


covmatrix=np.cov(df,bias=True)
print(covmatrix)
plt.figure(figsize=(20,20))
sns.heatmap(covmatrix, annot=True, fmt='g')
plt.show()


# In[17]:


sns.pairplot(df)
plt.show()


# In[ ]:





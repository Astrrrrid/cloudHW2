#!/usr/bin/env python
# coding: utf-8

# In[654]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
from scipy import stats


# In[655]:


import os
cardio = pd.read_csv("../Desktop/cardio_train2.csv",sep=";")
cardio.head()


# In[656]:


cardio=cardio.drop(columns=['id', 'active'])
cardio.head()


# In[657]:


cardio.columns = ['Age_days', 'Gender','Height_cm','Weight_kg','Systolic_Blood_Pressure',
                  'Diastolic_Blood_Pressure','Cholesterol','Glucose','Smoke','Alcohol','Cardiovascular_Disease']


# In[658]:


cardio.head()


# In[659]:


cardio.info()


# In[660]:


cardio.describe()


# ## filter out the outliers

# In[661]:



for x in cardio: 
    if x!="Gender" and x!="Age_days" and x!="Cholesterol" and x!="Glucose" and x!="Smoke" and x!="Alcohol"and x!="Cardiovascular_Disease" and x!="Obesity" and x!="Age_yrs"and x!="Age":
        q = cardio[x].quantile(0.99)
        q_low = cardio[x].quantile(0.01)
        q_hi  = cardio[x].quantile(0.99)
        cardio[x] = cardio[x][(cardio[x] < q_hi) & (cardio[x] > q_low)]


# In[ ]:





# In[662]:


cardio['Systolic_Blood_Pressure'] = cardio['Systolic_Blood_Pressure'].abs()
cardio['Diastolic_Blood_Pressure'] = cardio['Diastolic_Blood_Pressure'].abs()
cardio['Pulse_Pressure'] = cardio['Systolic_Blood_Pressure']-cardio['Diastolic_Blood_Pressure']
cardio['Pulse_Pressure'] = cardio['Pulse_Pressure'].abs()
cardio['BMI'] = cardio['Weight_kg']/((cardio['Height_cm']/100)*(cardio['Height_cm']/100))
cardio['Age_yrs']=cardio['Age_days']/365
#cardio['Age']=(cardio['Age_days'] / 365).round().astype('int')

cardio.head()


# In[663]:


def obes(c):
    if c['BMI']>=30:
        return "obesity"
    elif c['BMI']>=25:
        return "overweight"
    elif c['BMI']>=18.5:
        return "healthy"
    else:
        return "underweight"
    
cardio['Obesity']=cardio.apply(obes, axis=1)

cardio['Gender'].replace(2, 'Female',inplace=True)
cardio['Gender'].replace(1, 'Male',inplace=True)
cardio.head()


# In[664]:


cardio.dropna(inplace=True)
cardio.info()


# In[665]:


cardio.describe()


# ## 1. What is the age distribution for those who have cardiovascular disease? 

# In[666]:


df1 = cardio[cardio.Cardiovascular_Disease== 1]
df1.head()
#sns.countplot(x="Age", data=cardio)
#df['years'] = (df['age'] / 365).round().astype('int')
#sns.countplot(x='years', hue='cardio', data = df, palette="Set2")


# In[668]:


sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams['figure.dpi'] = 100
g1=sns.displot(x=df1.Age_yrs,bins=15,kde=True,color="c")


# ## 2. Does fitness have impact on the chance getting heart disease?

# In[669]:


sns.displot(x=cardio.BMI,bins=15,hue=cardio.Cardiovascular_Disease,kde=True,palette="GnBu")


# ## 3. Who is more likely getting the disease? 

# In[670]:


sex=df1.groupby('Gender').size() # df1 contains only cardiovascular diseases patients' info
sex=pd.DataFrame(sex)
sex

g5=sex.plot.pie(subplots=True,figsize=(4,4),autopct='%1.1f%%',shadow=True, startangle=55)


# ## 4. Obesity distribution among the patients 

# In[671]:


heal=df1.groupby('Obesity').size()
heal=pd.DataFrame(heal)
print(heal)


# In[685]:


g2=heal.plot.pie(subplots=True,figsize=(4,4),explode = (0.1, 0, 0, 0),autopct='%1.1f%%',shadow=True, startangle=35)


# ## 5. How does heart disease correlate with each variable? What are their relationships among the variables? 

# In[699]:


df3=cardio[['Height_cm',"Weight_kg","Cholesterol","Smoke",'Alcohol','Cardiovascular_Disease',
            "Glucose","Pulse_Pressure","Age_yrs"]]
df3=pd.DataFrame(df3)
df3['Cholesterol']=-df3['Cholesterol']
df3['Glucose']=-df3['Glucose']


# In[700]:


corr = df3.corr()
cmap=sns.color_palette("light:#5A9", as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ## 6. Pulse pressure analysis 

# In[704]:


def diff(c):
    if c['Pulse_Pressure']>40:
        return "dangerous"
    else:
        return "safe"
    
cardio['Hi_Lo']=cardio.apply(diff, axis=1)
cardio.head()
fig, ax = plt.subplots()
ax = sns.violinplot(y="Hi_Lo", x="BMI",data=cardio, inner="box",hue= "Cardiovascular_Disease", scale="width",split=True)
#fig.set_size_inches(11.7, 8.27)
sns.despine()


# In[702]:


#sns.violinplot(y="Weight_kg", x="Obesity",data=cardio, hue= "Gender", scale="width",split=True)


# In[691]:


#sns.set_theme(style="ticks")
#relation=cardio[['Age_yrs','BMI',"Cardiovascular_Disease"]]
#sns.pairplot(cardio, hue="Cardiovascular_Disease")


# ## 7. Fun fact for heights

# In[692]:


sns.kdeplot(shade=True,legend=True,x=cardio.Height_cm,hue=cardio.Gender)


# In[ ]:





# In[ ]:





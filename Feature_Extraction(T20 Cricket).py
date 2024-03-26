#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import numpy as np


# In[2]:


df = pickle.load(open("C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/dataset_level2.pkl", 'rb'))


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df[df['city'].isnull()]['venue'].value_counts()


# In[6]:


cities = np.where(df['city'].isnull(),df['venue'].str.split().apply(lambda x:x[0]),df['city'])


# In[7]:


df['city'] = cities


# In[8]:


df.isnull().sum()


# In[9]:


df.drop(columns = ['venue'],inplace=True)


# In[10]:


df.shape


# In[11]:


df['city'].value_counts()


# In[12]:


eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()


# In[13]:


df = df[df['city'].isin(eligible_cities)]


# In[14]:


df


# In[15]:


# Compute the cumulative sum of runs for each match_id and store the result in the 'current_score' column.
df['current_score'] = df.groupby('match_id').cumsum()['runs']


# In[16]:


df


# In[17]:


#Split the balls meas over value and ball value
df['over'] = df['ball'].apply(lambda x:str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x:str(x).split(".")[1])


# In[18]:


df


# In[19]:


#Find the number of a balls in one match
df['balls_bowled'] = (df['over'].astype('int')*6)+ df['ball_no'].astype('int')
df


# In[20]:


# Calculate remaining balls by subtracting balls bowled from 120, ensuring non-negativity, and update DataFrame 'df'.
df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x:0 if x<0 else x)
df


# In[21]:


#Find the wickets left
df['player_dismissed'] = df['player_dismissed'].apply(lambda x:0 if x == '0' else 1)
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['player_dismissed'] = df.groupby('match_id').cumsum()['player_dismissed']
df['wickets_left'] = 10 - df['player_dismissed']
df


# In[22]:


#Find the currunt runrate(crr)
df['crr'] = (df['current_score']*6)/df['balls_bowled']
df


# In[26]:


#Find the score of last 5 overs
groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=30).sum()['runs'].values.tolist())
last_five


# In[27]:


df['last_five'] = last_five


# In[28]:


df


# In[34]:


#Find the Total Score of First inning team
final_df = df.groupby('match_id').sum()['runs'].reset_index().merge(df,on='match_id')
final_df


# In[70]:


final_df[['batting_team','bowling_team','city','current_score','balls_left','wickets_left','crr','last_five','runs_x']]


# In[ ]:


final_df.dropna(inplace=True)


# In[ ]:


final_df.isnull().sum()


# In[ ]:


final_df = final_df.sample(final_df.shape[0])


# In[ ]:


final_df.sample(2)


# In[ ]:


X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


X_train


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error


# In[ ]:


trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[ ]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3',XGBRegressor(n_estimators=1000,learning_rate=0.2,max_depth=12,random_state=1))
])


# In[ ]:


pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


# In[ ]:


pipe = pickle.load(open('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/pipe.pkl', 'wb'))


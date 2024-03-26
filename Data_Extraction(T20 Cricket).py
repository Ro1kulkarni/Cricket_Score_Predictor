#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np      # For numerical computations
import pandas as pd     # For data manipulation and analysis
from yaml import safe_load   # For parsing YAML files
import os               # For interacting with the operating system
from tqdm import tqdm  # For creating progress bars


# In[3]:


filenames = []
for file in os.listdir('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/Data'):
    filenames.append(os.path.join('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/Data',file))


# In[4]:


filenames[:5]


# In[33]:


# Combine data from multiple JSON files into a single DataFrame, adding a match_id column to identify each match.
final_df = pd.DataFrame()
counter = 1
for file in tqdm(filenames):
    with open(file, 'r') as f:
        df = pd.json_normalize(safe_load(f))
        df['match_id'] = counter
        final_df = final_df.append(df)
        counter += 1
final_df


# In[34]:


final_df


# In[35]:


backup = final_df


# In[36]:


final_df.columns


# In[39]:


final_df.drop(columns=[
    'meta.data_version',
    'meta.created',
    'meta.revision',
    'info.outcome.bowl_out',
    'info.bowl_out',
    'info.supersubs.South Africa',
    'info.supersubs.New Zealand',
    'info.outcome.eliminator',
    'info.outcome.result',
    'info.outcome.method',
    'info.neutral_venue',  
    'info.match_type_number',
    'info.outcome.by.runs',
    'info.outcome.by.wickets'
],inplace=True)


# In[40]:


final_df


# In[41]:


final_df.columns


# In[42]:


final_df['info.gender'].value_counts()


# In[43]:


final_df = final_df[final_df['info.gender'] == 'male']
final_df.drop(columns = ['info.gender'],inplace=True)
final_df


# In[44]:


final_df['info.match_type'].value_counts()


# In[46]:


final_df['info.overs'].value_counts()


# In[47]:


final_df = final_df[final_df['info.overs'] == 20]
final_df.drop(columns=['info.overs', 'info.match_type'], inplace = True)
final_df


# In[48]:


import pickle
pickle.dump(final_df,open('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/dataset_level.pkl', 'wb'))


# In[49]:


matches = pickle.load(open('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor/dataset_level.pkl', 'rb'))
matches


# In[52]:


matches.iloc[0]['innings'][0]['1st innings']['deliveries']


# In[90]:


# Iterate through each match, extract ball-by-ball data for the 1st innings, and append it to a DataFrame.
count = 1
delivery_df = pd.DataFrame()
for index, row in matches.iterrows():
    if count in [75,108,150,180,268,360,443,458,584,748,982,1052,1111,1226,1345]:
        count += 1
        continue
    count += 1
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []
    match_id = []
    city = []
    venue = []
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            match_id.append(count)
            batting_team.append(row['innings'][0]['1st innings']['team'])
            teams.append(row['info.teams'])
            ball_of_match.append(key)
            batsman.append(ball[key]['batsman'])
            bowler.append(ball[key]['bowler'])
            runs.append(ball[key]['runs']['total'])
            city.append(row['info.city'])
            venue.append(row['info.venue'])
            try:
                player_of_dismissed.append(ball[key]['wicket']['player_out'])
            except:
                player_of_dismissed.append('0')
    loop_df = pd.DataFrame({
        'match_id':match_id,
        'teams':teams,
        'batting_team':batting_team,
        'ball':ball_of_match,
        'batsman':batsman,
        'bowler':bowler,
        'runs':runs,
        'player_dismissed':player_of_dismissed,
        'city':city,
        'venue':venue
    })
    delivery_df = delivery_df.append(loop_df)
            


# In[91]:


delivery_df


# In[98]:


backup1 = delivery_df


# In[92]:


#It gives Bowling Team
def bowl(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team


# In[93]:


delivery_df['bowling_team'] = delivery_df.apply(bowl,axis=1)


# In[94]:


delivery_df


# In[95]:


delivery_df.drop(columns=['teams'],inplace=True)


# In[96]:


delivery_df


# In[97]:


delivery_df['batting_team'].unique()


# In[99]:


teams = [
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka'
]


# In[100]:


delivery_df = delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_df = delivery_df[delivery_df['bowling_team'].isin(teams)]


# In[101]:


delivery_df


# In[103]:


output = delivery_df[['match_id','batting_team','bowling_team','ball','runs','player_dismissed','city','venue']]


# In[104]:


output


# In[106]:


pickle.dump(output,open('C:/Users/Hrishikesh/Desktop/rohan/T20 World Cup Cricket Score Predictor//dataset_level2.pkl', 'wb'))


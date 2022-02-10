#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, db
import json
import random


# In[2]:


#cred = credentials.Certificate('/Users/evanrussek/server_keys/feature_task_key.json')
cred = credentials.Certificate('C:\\Users\\User\\Documents\\python\\Feature_learning_anxiety\\pilot_r1\\feature_task_key.json')
default_app = firebase_admin.initialize_app(cred)
client = firestore.client()


# In[4]:


dfs = []
for subj in client.collection('featuretask').document('run1b').collection('subjects').stream():
    taskdata_collection = client.collection('featuretask/run1b/subjects/{0}/taskdata'.format(subj.id)).stream()
    subjectID = subj.id
    has_data = False # did this subject finish the task?\n",
    has_start = False
    for tc in taskdata_collection:
        if tc.id == "start":
            has_start = True
            start_dict = tc.to_dict()
            subjectID = start_dict['subjectID']    
        else:
            has_data = True
            task_dict = tc.to_dict()
            sub_data = json.loads(task_dict['data'])
            sub_df = pd.DataFrame(sub_data)
            #sub_df['subjectID'] = subj.id
        
    if has_start & has_data:
        sub_df["subjectID"] = subjectID
        dfs.append(sub_df)

combined_df = pd.concat(dfs, sort = True)
combined_df.to_csv('data/run1b_data.csv')



    


# In[6]:


# get one subject dfs
bonus_df = pd.DataFrame()
print (combined_df['subjectID'].str.split(';\s*', expand=True).stack().unique())
subj_IDs = combined_df['subjectID'].str.split(';\s*', expand=True).stack().unique()
for i in range(len(subj_IDs)):
    s1_data = combined_df[combined_df["subjectID"] == subj_IDs[i]]
    rr = s1_data.reward_received[~np.isnan(s1_data.reward_received)][5:]
    bonus_points = np.sum(rr.sample(5))
    bonus_pct = (bonus_points + 15)/30
    bonus = 2*bonus_pct
    bonus_dict = {'subjectID': subj_IDs[i], 'bonus': bonus}
    bonus_df = bonus_df.append(bonus_dict, ignore_index = True)

bonus_df[['subjectID', 'bonus']].to_csv('bonus_files/run1b_bonus_payments.csv', sep = ',', header = False, index = False)


# In[61]:


i = 0
s1_data = combined_df[combined_df["subjectID"] == subj_IDs[i]]


# In[62]:


rr = s1_data.reward_received[~np.isnan(s1_data.reward_received)][5:]
bonus_points = np.sum(rr.sample(5))
bonus_pct = (bonus_points + 15)/30
bonus = 2*bonus_pct
bonus_dict = {'subjectID': subj_IDs[i], 'bonus': bonus}


# # In[63]:


print(bonus_dict)





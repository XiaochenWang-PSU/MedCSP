#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, sys, time
from datetime import datetime, timedelta
import pickle

from collections import Counter


# In[2]:


import yaml
config = yaml.safe_load(open('../config.yaml'))
data_path = config['data_path']
mimic3_path = config['mimic3_path']

import pathlib
pathlib.Path(data_path, 'population').mkdir(parents=True, exist_ok=True)


# In[3]:


patients = pd.read_csv(mimic3_path + 'combined_pat.csv', parse_dates=['DOB', 'DOD'], usecols=['SUBJECT_ID', 'DOB', 'DOD'])
admissions = pd.read_csv(mimic3_path + 'combined_adm.csv', parse_dates=['DEATHTIME'], usecols=['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])
examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(by='ICUSTAY_ID') # Only Metavision

examples = pd.merge(examples, patients, on='SUBJECT_ID', how='left')
examples = pd.merge(examples, admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
examples['AGE'] = examples.apply(lambda x: (x['INTIME'].to_pydatetime() - x['DOB'].to_pydatetime()).total_seconds(), axis=1) / 3600 / 24 / 365.25

examples['LOS'] = examples['LOS'] * 24 # Convert to hours


# In[4]:


tasks = ['ARF', 'Shock', "Pretrain"]
label_defs = { task: pd.read_csv(data_path + 'labels/{}.csv'.format(task)) for task in tasks }


# In[5]:


# Start
N = len(examples['ICUSTAY_ID'].unique())
print('Source population', N)


# In[6]:


#assert (examples['INTIME'] <= examples['OUTTIME']).all()
#assert (examples['DBSOURCE'] == 'metavision').all()


# In[7]:


# Remove non-adults
#min_age = 18
#max_age = np.inf # no max age
#examples = examples[(examples.AGE >= min_age) & (examples.AGE <= max_age)]
#print('Exclude non-adults', examples['ICUSTAY_ID'].nunique())
examples_ = examples


# In[8]:


for T in [48.0]:#, 12.0, 48.0]:
    print('======')
    print('prediction time', T, 'hour')
    print(len(examples))
    # Remove died before cutoff hour
    examples = examples_[(examples_.DEATHTIME >= examples_.INTIME + timedelta(hours=T)) | (examples_.DEATHTIME.isnull())]
    print('Exclude deaths', examples['ICUSTAY_ID'].nunique())

    # Remove LOS < cutoff hour
    examples = examples[examples['LOS'] >= T]
    print('Exclude discharges', examples['ICUSTAY_ID'].nunique())

    populations = {}
    # Remove event onset before (cutoff)
    for task in tasks:
        if task != "Pretrain":
            print('---')
            print('Outcome', task)
            label_def = label_defs[task]
    
            # Needed to preserve index in DataFrame
            print(len(examples), len(label_def))
            pop = examples[['ICUSTAY_ID']].reset_index() \
                    .merge(label_def[['ICUSTAY_ID', '{}_ONSET_HOUR'.format(task)]], on='ICUSTAY_ID', how='left') \
                    .set_index('index').copy()
            pop = pop[(pop['{}_ONSET_HOUR'.format(task)] >= T) | pop['{}_ONSET_HOUR'.format(task)].isnull()]
            
            pop['{}_LABEL'.format(task)] = pop['{}_ONSET_HOUR'.format(task)].notnull().astype(int)
            pop = pop.rename(columns={'ICUSTAY_ID': 'ID'})
            pop.to_csv(data_path + 'population/{}_{}h.csv'.format(task, T), index=False)
    
            # Construct boolean mask
            ## NOTE: uses pop.index here, assuming index is preserved
            idx = pop.index
            ## Otherwise, there's a slower version
            # if False:
            #    idx = np.array([examples[examples.ICUSTAY_ID == i].index[0] for i in pop['ICUSTAY_ID']])
            mask_array = np.zeros(N, dtype=bool)
            mask_array[idx] = True
    
            # Save population boolean mask
            np.save(data_path + 'population/mask_{}_{}h.npy'.format(task, T), mask_array)
            np.savetxt(data_path + 'population/mask_{}_{}h.txt'.format(task, T), mask_array, fmt='%i')
    
            populations[task] = pop
            print('Exclude onset', len(pop))
        else:
            print('---')
            print('Outcome', task)
            label_def = label_defs[task]
            label_def = label_def.rename(columns={'ICUSTAY_ID': 'ID'})
            print(len(examples), len(label_def))
            label_def.to_csv(data_path + 'population/{}_{}h.csv'.format(task.lower(), T), index=False)
            
# In[9]:


for T in [48.0]:
    print('======')
    print('prediction time', T, 'hour')

#     # Remove died before cutoff hour
    examples = examples_[(examples_.DEATHTIME >= examples_.INTIME + timedelta(hours=T)) | (examples_.DEATHTIME.isnull())]
    print('Exclude deaths', examples['ICUSTAY_ID'].nunique())

#     # Remove LOS < cutoff hour
    examples = examples[examples['LOS'] >= T]
    print('Exclude discharges', examples['ICUSTAY_ID'].nunique())

#     # Remove event onset before (cutoff)
    for task in ['mortality']:
        print('---')
        print('Outcome', task)
        examples['{}_LABEL'.format(task)] = examples.HOSPITAL_EXPIRE_FLAG
        pop = examples[['ICUSTAY_ID', '{}_LABEL'.format(task)]]
        pop = pop.rename(columns={'ICUSTAY_ID': 'ID'})
        pop.to_csv(data_path + 'population/{}_{}h.csv'.format(task, T), index=False)
        print('Exclude onset', len(pop))


# In[ ]:





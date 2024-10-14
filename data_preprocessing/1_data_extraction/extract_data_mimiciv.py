"""
extract_data.py
Author: Shengpu Tang

Cleans each structured table in MIMIC-III and converts it into a standard format 
containing the following 4 columns: 

    –––––––––––––––––––––––––––––––––––––––––––––––––––
    |  ID  |  t  |  variable_name  |  variable_value  |
    –––––––––––––––––––––––––––––––––––––––––––––––––––

If a row has t = 'NULL', then the information is assumed to be time-invariant 
and always available.

Note:
    - Estimated runtime: 30 min
    - To execute the entire script, run:
        `python prepare.py`
    - To execute a single function, run:
        `python -c "from prepare import *; merge_items_table();"`

List of structured tables in MIMIC-III that are used:
    - PATIENTS
    - ADMISSIONS
    - ICUSTAYS
    - CHARTEVENTS
    - LABEVENTS
    - MICROBIOLOGYEVENTS
    - OUTPUTEVENTS
    - INPUTEVENTS_MV
    - PROCEDUREEVENTS_MV
    - DATETIMEEVENTS
"""

def main():
    merge_items_table()
    extract_icustays()
    
    # Time-invariant data: PATIENTS, ADMISSIONS, ICUSTAYS
    extract_invariant()
    
    # Time-series data: discrete events
    extract_events('labevents', 'charttime', ['value', 'FLAG'])
    extract_events('microbiologyevents', 'charttime')
    extract_events('datetimeevents', 'charttime', ['value'])
    extract_events('outputevents', 'charttime', ['value'])
    extract_events('chartevents', 'charttime', ['value'], 2000000)
    
    # Time-series data: continuous events
    # extract_events_continuous(
        # 'combined_ipt', 'STARTTIME', 'ENDTIME', 
        # ['ORDERCATEGORYDESCRIPTION', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM']
    # )
    # extract_events_continuous('combined_pcd', 'STARTTIME', 'ENDTIME')
    
    # Additional processing & formatting
    # convert_inputevents_units()
    
    data = stack_attr_columns()


################################
####    Helper functions    ####
################################

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
import pickle
import yaml

from config import config, data_path, mimic4_path, parallel

config['n_rows'] = {
    'CHARTEVENTS': 330712483,
    'DATETIMEEVENTS': 4485937,
    'INPUTEVENTS_MV': 3618991,
    'LABEVENTS': 27854055,
    'MICROBIOLOGYEVENTS': 631726,
    'OUTPUTEVENTS': 4349218,
    'PROCEDUREEVENTS_MV': 258066,
}
config['n_itemids'] = 4181
config['n_icustays'] = 23620

import pathlib
pathlib.Path(data_path, 'prep').mkdir(parents=True, exist_ok=True)
pathlib.Path(data_path, 'formatted').mkdir(parents=True, exist_ok=True)

def print_header(*content, char='-'):
    print()
    print(char * 80)
    print(' ' * 8, *content)
    print(char * 80)

def check_nrows():
    """
    Verify the integrity of raw csv files.
    """
    for fname, n_rows in config['n_rows'].items():
        print(fname, '...')
        actual_rows = 0
        for df in pd.read_csv(mimic4_path + '{}.csv'.format(fname), chunksize=10000000):
            actual_rows += len(df)
        assert n_rows == actual_rows, 'Expected {}, got {}'.format(n_rows, actual_rows)

def merge_items_table():
    """
    Combine D_ITEMS and D_LABITEMS, keeping only ITEMIDs from metavision and hospital
    (Removing carevue ITEMIDs)
    
    Returns a table that contains all the ITEMIDs (sorted) that will be used. 
    Saves this table to {data_path}/items_table.csv
    
    1-8572: chartevents (carevue), datetimeevents (carevue)
    30001-30405: inputevents_cv
    40030-46807: inputevents_cv (carevue), outputevents (carevue)
    50800-51555: labevents
    70001-70093: Microbiology SPECIMEN
    80001-80312: Microbiology ORGANISM
    90001-90031: Microbiology ANTIBACTERIUM
    220003-228647: everything for metavision, i.e.
        [chartevents, outputevents, inputevents_mv, procedureevents_mv, datetimeevents_mv]
    """
    print_header('Merging ITEMID tables')
    
    d_items = pd.read_csv(mimic4_path + 'icu/d_items.csv')
    d_labitems = pd.read_csv(mimic4_path + 'hosp/labevents.csv')
    d_labitems['LINKSTO'] = 'labevents'

    items_table = pd.concat(
        [d_items, d_labitems], axis=0, ignore_index=True, sort=False
    )
    items_table.sort_values('itemid', inplace=True)
    items_table.to_csv(data_path + 'prep/items_table.csv', index=False)
    # assert len(items_table) == config['n_itemids']
    print('Done!')
    return items_table

def extract_icustays():
    """
    Reads the ICUSTAYS table which contains information about each ICU admission 
    and the corresponding hospital admission and patient ID.
    Splits ICU stays into train/val/test at the patient level. 
    
    Returns a table that holds information about the Metavision ICU admissions
    """
    print_header('Extracting ICU stays')
    
    def read_icustays_table():
        df = pd.read_csv(
            mimic4_path + 'icu/icustays.csv',
            parse_dates=['intime', 'outtime'],
        )
        df.sort_values(by='subject_id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    df_icu = read_icustays_table()
    print(df_icu.columns)
    df_icu = df_icu.sort_values(by='icustay_id').reset_index(drop=True)
    df_icu.to_csv(data_path + 'prep/icustays_MV.csv', index=False)
    return df_icu_MV

def extract_invariant():
    """
    Returns a table that contains data available at the time of ICU admission. 
    Saves table to as {data_path}/invariant.csv
    
    Patients: 
        'GENDER', Age (calculated from 'DOB')
    
    Admission: 
        'ADMITTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 
        'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY'
    
    ICU stays: 
        'FIRST_CAREUNIT', 'FIRST_WARDID', HADM_to_ICU_time
    """
    print_header('Extracting time-invariant data')
    
    patients = pd.read_csv(
        mimic4_path + 'hosp/patients.csv',  usecols=['subject_id', 'gender', 'anchor_age']
    )
    
    admissions = pd.read_csv(
        mimic4_path + 'hosp/admissions.csv', 
        parse_dates=['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'],
    )
    admissions = admissions[[
        'subject_id', 
        'hadm_id', 'admittime', 'admission_type', 'admission_location', 'insurance', 
        'language',  'marital_status'
    ]]
    # dropped columns=['DIAGNOSIS', 'DISCHTIME', 'DEATHTIME', 'DISCHARGE_LOCATION', 'EDREGTIME', 'EDOUTTIME', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA']
    
    examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['intime', 'outtime'])
    examples = examples[['subject_id', 'hadm_id', 'icustay_id', 'firstcare_unit',  'intime']]
    
    invariant = pd.merge(examples, patients, on='subject_id', how='left')
    invariant = pd.merge(invariant, admissions, on=['subject_id', 'hadm_id'], how='left')
    
    invariant['AGE'] = invariant.ANCHOR_AGE
    
    invariant['HADM_to_ICU_time'] = \
        (invariant['intime'] - invariant['admittime']).apply(lambda x: x.total_seconds()) / 3600  # How long in hospital before admitted to ICU
    invariant = invariant.drop(columns=['subject_id', 'hadm_id', 'intime', 'admittime'])
    invariant = invariant.rename(columns={'icustay_id': 'ID'}).set_index('ID')
    invariant.to_csv(data_path + 'prep/invariant.csv')
    # assert len(invariant) == config['n_icustays']
    print('Done!')
    return invariant

def _read_events(fname, t_cols, chunksize):
    """
    A helper function to read csv in chunks
    Arguments:
        - fname is the file name (i.e INPUTEVENTS)
        - t_cols is a list that contains the names of the time columns that should be parsed
        - chunksize is the size of each chunk
    """
    n_rows = 1e8
    with tqdm(desc=fname, total=(n_rows//chunksize+1)) as pbar:
        for df in pd.read_csv(mimic4_path + '{}.csv'.format(fname), parse_dates=t_cols, chunksize=chunksize):#, usecols = ['VALUENUM', 'STORETIME', 'WARNING', 'ITEMID', 'VALUE','CHARTTIME', 'ICUSTAY_ID', 'VALUEUOM']):
            pbar.update()
            yield df

def _process_chunk(df, fname):
    """
    Makes adjustments (e.g. making the units consistent, removing invalid instances) 
    to each chunk that is passed in based on which table it came from.
    Returns the modified chunk.
    Arguments:
        - df is a 'chunk' of data that is made by the _read_events function
        - fname is the file name (i.e INPUTEVENTS)
    """
    # When there is no ICUSTAY_ID, each event is matched by subject_id and hadm_id, 
    # and subsequently filtered by time when we define the period of interest
    if 'icustay_id' not in df.columns:
        df_icu = pd.read_csv(mimic4_path + 'icu/icustays.csv', parse_dates=['intime', 'outtime']).sort_values(by='subject_id').reset_index(drop=True)
        df = pd.merge(df_icu[['subject_id', 'hadm_id', 'icustay_id']], df, on=['subject_id', 'hadm_id'], how='inner')
        df = df.drop(columns=['subject_id', 'hadm_id'])
    
    if fname.lower() == 'microbiologyevents':
        # Remove null specimen
        df = df.dropna(subset=['micro_specimen_id'])
        
        # Consider only type of specimen collected
        df = df.rename(columns={'micro_specimen_id': 'ITEMID'})
        df['ITEMID'] = df['ITEMID'].astype(int)
    
        # Check each itemid has either CHARTTIME or CHARTDATE
        itemid_date = df[pd.isnull(df['charttime'])]['ITEMID'].unique()
        itemid_time = df[~pd.isnull(df['charttime'])]['ITEMID'].unique()
        itemid_keep = set(itemid_time) - set(itemid_date)
        
        # Keep only rows with chart time
        df = df.dropna(subset=['charttime'])
    
    if fname.lower() == 'inputevents':
        # Remove rewritten entries
        # https://mimic.physionet.org/mimictables/inputevents_mv/#statusdescription

        
        # Remove zero or negative amount
        # https://github.com/MIT-LCP/mimic-code/issues/464
        df = df.loc[df.amount > 0.0]
        
        # Wrong units
        df = df.loc[df.amountuom != '/hour']
        
        # Calculate rate for Additives (Continuous IV)
        ## Originally these rows don't have rate, only amount
        norate = pd.isnull(df['rate'])
        df_norate = df[norate].copy()
        ## Identify Additives by their route
        additives = df_norate['ordercategorydescription'] == 'Continuous IV'
        df_additives = df_norate[additives].copy()
        assert df_additives['secondaryordercategoryname'].str.contains('Additive').all()
        ## Duration could be >= 1min
        duration = (df_additives['endtime'] - df_additives['starttime']).dt.total_seconds() / 60. # in minutes
        ## Set the values in the original dataframe
        df_norate.loc[additives, 'rate'] = df_additives['amount'] / duration
        df_norate.loc[additives, 'rateuom'] = df_additives['amountuom'].apply(lambda s: s + '/min')
        df.loc[norate, :] = df_norate
    

    
    if fname.lower() == 'chartevents':


        d_items = pd.read_csv(data_path + 'prep/items_table.csv')
        chart_items = d_items[d_items.CATEGORY != 'Labs']
        
        if len(df) > 0:
            df['valueuom'] = df['valueuom'].astype(str)
            vitals_map = yaml.full_load(open('grouped_variables.yaml'))
            
            # Convert temperature units
            temperature_items = vitals_map['Temperature']
            fahrenheit = (df.itemid.isin(temperature_items)) & (df.valueuom.astype(str).str.endswith('F'))
            df.loc[fahrenheit, 'value'] = df.loc[fahrenheit, 'value'].apply(lambda x: (float(x) - 32) / 1.8)
            df.loc[fahrenheit, 'valueuom'] = '?C'
            assert (df.loc[(df.ITEMID.isin(temperature_items)), 'valueuom'].str.endswith('C')).all()
            df.loc[(df.itemid.isin(temperature_items)), 'valueuom'] = '?C'
            
            # Convert weight units
            weight_items = vitals_map['Weight'] # [224639, 226512, 226531]
            lbs = (df.itemid.isin(weight_items)) & (df.VALUEUOM.astype(str) != 'kg')
            df.loc[lbs, 'value'] = df.loc[lbs, 'value'].apply(lambda x: float(x) / 2.2046226218)
            df.loc[lbs, 'valueuom'] = 'kg'
            assert (df.loc[(df.itemid.isin(weight_items)), 'valueuom'] == 'kg').all()
            
            # Convert height units
            height_items = vitals_map['Height'] # [226707, 226730]
            inch = (df.itemid.isin(height_items)) & (df.valueuom.astype(str) == 'Inch')
            df.loc[inch, 'value'] = df.loc[inch, 'value'].apply(lambda x: float(x) / 0.3937008)
            df.loc[inch, 'valueuom'] = 'cm'
            assert (df.loc[(df.itemid.isin(height_items)), 'valueuom'] == 'cm').all()
    return df

def _finalize_table(df, fname):
    """
    Makes final adjustments to the chartevents and datetime events tables:
    + chartevents: replace grouped ITEMIDs with names
    + datetimeevents: convert timestamps to days relative to ICU admission time
    """
    # converting ITEMID and ICUSTAY_ID to integers
    # df = df.copy()
    df['ITEMID'] = df['ITEMID'].astype(int)
    df['ICUSTAY_ID'] = df['ICUSTAY_ID'].astype(int)
    
    # Dropping the rows where end_time is recorded as occuring before the start_time
    if 't_start' in df.columns and 't_end' in df.columns:
        df = df[df['t_start'] <= df['t_end']]
    
    if fname.lower() == 'chartevents':
        # Replace vital itemid with names
        vitals_map = yaml.full_load(open('grouped_variables.yaml'))
        items = sum(vitals_map.values(), [])
        items_map = {v:k for k in vitals_map for v in vitals_map[k]}
        df['ITEMID'] = df['ITEMID'].replace(items_map)
    
    if fname.lower() == 'datetimeevents':
        # Convert timestamps to numbers
        examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['intime', 'outime'])
        df['VALUE'] = pd.to_datetime(df['value'],errors = "coerce")
        print(df['value'])
        df = df[df["value"].notna()]
        # df['VALUE'] = df['VALUE'].apply(np.datetime64)
        df = pd.merge(df, examples[['icustay_id', 'initime']], on='icustay_id', how='left')
        
        # Convert time interval to fractional days, value - intime
        ## Doing manual for-loop to prevent integer overflow in pd.Timestamp
        print('*** Converting Timestamp to float')
        print('    (time interval relative to INTIME, in fractional days)')
        # values = [ 
        values =    (df['value'] - df['intime']).apply(lambda x: x.total_seconds() / (3600. * 24.)) 
            # (df['VALUE'][i] - np.datetime64(df['INTIME'][i])).item().total_seconds() / (3600. * 24.) 
            # for i in tqdm(range(len(df)), desc='DATETIMEEVENTS_conversion')
        # ]
        df['value'] = values
    
    if fname.lower() == 'microbiologyevents':
        # Remove repeated rows since we only care about what specimen was drawn
        # and each specimen could have several results recorded
        df.drop_duplicates(subset=['icustay_id', 't', 'itemid'], inplace=True)
    
    df['itemid'] = df['itemid'].astype(str)
    if 't' in df.columns:
        df = df.sort_values(by=['icustay_id', 't']).reset_index(drop=True)
    else:
        assert 't_start' in df.columns
        df = df.sort_values(by=['icustay_id', 't_start', 't_end']).reset_index(drop=True)
    
    if fname.lower() == 'inputevents':
        df.to_pickle(data_path + 'prep/{}.raw.p'.format('combined_ipt'.lower()))
    return df

def extract_events(fname, t_col, val_cols=[], chunksize=500000):
    """
    Prepares the tables by calling the _process_chunk function.
    This function is used for point-time events that only charttime.
    Saves the prepared table as a pickle file: filename.p
    Arguments:
        - fname: name of file to be read (i.e CHARTEVENTS)
        - t_col: time column (i.e charttime)
        - val_cols: other columns to be extracted (i.e value, route (if available))
    """
    print_header('Extracting events data from', fname)
    
    examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['intime', 'outtime'])
    d_items = pd.read_csv(data_path + 'prep/items_table.csv')
    
    df = pd.DataFrame()
    for df_chunk in _read_events(fname, [t_col], chunksize):
        df_chunk = _process_chunk(df_chunk, fname)
        
        # Filter patients, keep only Metavision
        df_chunk = pd.merge(df_chunk, examples, on='icustay_id', how='inner')
        
        # Filter item_ids, keep only Metavision
        df_chunk = pd.merge(df_chunk, d_items[['itemid']], on='itemid', how='inner')
        
        # Calculate t
        df_chunk['t'] = (df_chunk[t_col].dt.total_seconds() - df_chunk['intime'].dt.total_seconds()) / 3600. # Hours
        
        df = df.append(df_chunk[['icustay_id', 't', 'itemid'] + val_cols])
    print("start finalize")
    df = _finalize_table(df, fname)  
    df.to_pickle(data_path + 'prep/{}.p'.format(fname.lower()))
    print('Done!')
    return df

def extract_events_continuous(fname, t_start, t_end, val_cols=[], chunksize=1000000):
    """
    Prepares the tables by calling the _process_chunk function.
    This function is used for range-time events that have t_start and t_end columns
    Saves the prepared table as a pickle file: filename.p
    Arguments:
        - fname: name of file to be read (i.e CHARTEVENTS)
        - t_start: start time column
        - t_end: end time column
        - val_cols: other columns to be extracted (i.e value, route (if available))
    """
    print_header('Extracting events data from', fname)
    
    examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['intime', 'outtime'])
    d_items = pd.read_csv(data_path + 'prep/items_table.csv')
    
    df = pd.DataFrame()
    for df_chunk in _read_events(fname, [t_start, t_end], chunksize):
        df_chunk = _process_chunk(df_chunk, fname)
        
        # Filter patients, keep only Metavision
        df_chunk = pd.merge(df_chunk, examples, on='icustay_id', how='inner')
        
        # Filter item_ids, keep only Metavision
        df_chunk = pd.merge(df_chunk, d_items[['itemid']], on='itemid', how='inner')
        
        # Calculate 
        # to do: for mimic iv, use end-start to get timestamp
        
        df_chunk['t_start'] = (df_chunk[t_start] - df_chunk['intime']).dt.total_seconds() / 3600. # Hours
        df_chunk['t_end'] = (df_chunk[t_end] - df_chunk['intime']).dt.total_seconds() / 3600. # Hours
        
        df = df.append(df_chunk[['icustay_id', 't_start', 't_end', 'itemid'] + val_cols])
    
    df = _finalize_table(df, fname)
    df.to_pickle(data_path + 'prep/{}.p'.format(fname.lower()))
    print('Done!')
    return df

def convert_inputevents_units():
    """
    This function makes all the units in the inputevents table consistent
    - converts mass to mg
    - converts volume to mL
    - converts 'dose' units to NAN (treat 'dose' as its own special unit)
    - calculates rate for continuous infusions
    """
    from collections import Counter
    print_header('Converting units (INPUTEVENTS_MV)')
    
    fname = 'inputevents'
    df = pd.read_pickle(data_path + 'prep/{}.raw.p'.format(fname.lower()))
    
    # Consistency check: No rate, then no rate unit
    assert (pd.isnull(df['rate']) == pd.isnull(df['rateuom'])).all()
    
    # Consistency check: No amount, then no amount unit
    assert (pd.isnull(df['amount']) == pd.isnull(df['amountuom'])).all()
    
    # Handle one-off / continual infusion separately
    df_norate = df[pd.isnull(df['rateuom'])]
    df_rate = df[~pd.isnull(df['rateuom'])]
    df_rate = df_rate[df_rate.AMOUNTUOM != 'dose'] # Erroneous unit for continuous infusion

    # One-off infusion
    drugs_norate = {}
    for itemid, group in df_norate.groupby('itemid'):
        group = group.copy()

        # Mass
        mcg = (group['amountuom'] == 'mcg')
        group.loc[mcg, 'amount'] = group.loc[mcg, 'AMOUNT'] * 0.001
        group.loc[mcg, 'amountuom'] = 'mg'
        grams = (group['amountuom'] == 'grams')
        group.loc[grams, 'amount'] = group.loc[grams, 'AMOUNT'] * 1000.0
        group.loc[grams, 'amountuom'] = 'mg'

        # Volume
        L = (group['amountuom'] == 'L')
        group.loc[L, 'amount'] = group.loc[L, 'AMOUNT'] * 1000.0
        group.loc[L, 'amountuom'] = 'ml'
        uL = (group['amountuom'] == 'uL')
        group.loc[uL, 'amount'] = group.loc[uL, 'AMOUNT'] * 0.001
        group.loc[uL, 'amountuom'] = 'ml'
        group.loc[group['amountuom'] == 'ml', 'AMOUNTUOM'] = 'mL'

        # Dose
        dose = (group.AMOUNTUOM == 'dose')
        group.loc[dose, 'dose'] = group.loc[dose, 'amount']
        group.loc[dose, 'amount'] = np.nan
        group.loc[dose, 'amountuom'] = np.nan

        units = set(group['amountuom'].unique())
        if (len(units) == 1): # only one unit
            print(itemid, units, end='\r', flush=True)
        elif len(units) > 1:
            print(itemid, Counter(group['AMOUNTUOM']), end='\r', flush=True)
        else:
            assert False
        drugs_norate[itemid] = group

    # Continual Infusion
    drugs_rate = {}
    for itemid, group in df_rate.groupby('ITEMID'):
        group = group.copy()
        # print(itemid, end='\r', flush=True)

        # Mass units -> mg
        mcg = (group['amountuom'] == 'mcg')
        group.loc[mcg, 'amount'] = group.loc[mcg, 'AMOUNT'] * 0.001
        group.loc[mcg, 'amountuom'] = 'mg'
        grams = (group['amountuom'] == 'grams')
        group.loc[grams, 'amount'] = group.loc[grams, 'AMOUNT'] * 1000.0
        group.loc[grams, 'amountuom'] = 'mg'
        pg = (group['amountuom'] == 'pg')
        group.loc[pg, 'amount'] = group.loc[pg, 'AMOUNT'] * 1e-9
        group.loc[pg, 'amountuom'] = 'mg'

        # Volume
        group.loc[group['amountuom'] == 'ml', 'AMOUNTUOM'] = 'mL'
        litter = (group['amountuom'] == 'L')
        group.loc[litter, 'amount'] = group.loc[litter, 'AMOUNT'] * 1000.0
        group.loc[litter, 'amountuom'] = 'mL'
        cm3 = (group['amountuom'] == 'cm3')
        group.loc[cm3, 'amountuom'] = 'mL'
        mm3 = (group['amountuom'] == 'mm^3')
        group.loc[mm3, 'amount'] = group.loc[mm3, 'AMOUNT'] / 1000.0
        group.loc[mm3, 'amountuom'] = 'mL'
        
        ounce = (group['amountuom'] == 'ounces')
        group.loc[ounce, 'amount'] = group.loc[ounce, 'AMOUNT'] * 28349.5
        group.loc[ounce, 'amountuom'] = 'mg'
        
        group.loc[group['amountuom'] == 'mEq.', 'AMOUNTUOM'] = 'mEq'
        group.loc[group['amountuom'] == 'ml/hr', 'AMOUNTUOM'] = 'mEq'
        
        
        group.loc[group['amountuom'] == 'International Units', 'AMOUNTUOM'] = 'units'
        group.loc[group['amountuom'] == 'nMol/ml/min', 'AMOUNTUOM'] = 'mEq'
        uniques = group['amountuom'].nunique()
        
        print(group['amountuom'].unique())
        if (uniques != 1):
            # print(group.loc[group["AMOUNTUOM"] == 'mEq.']["AMOUNT"])#['AMOUNTUOM'].unique())
            # print(group.loc[group["AMOUNTUOM"] == 'ml/hr']["AMOUNT"])
            raise("what's up")
            # raise (itemid, Counter(group['AMOUNTUOM']))

        # Calculate rate
        group['rate'] = group.AMOUNT / ((group.t_end - group.t_start) * 60)
        group['rateunit'] = group.AMOUNTUOM.apply(lambda s: s + '/min')
        group['amount'] = np.nan
        group['amountuom'] = np.nan

        drugs_rate[itemid] = group

    # Combine data tables
    df = pd.concat([*drugs_rate.values(), *drugs_norate.values()], axis=0, ignore_index=True, sort=False)
    df = df[['icustay_id', 't_start', 't_end', 'itemid', 'ordercategorydescription', 'amount', 'amountuom', 'rate', 'rateunit', 'dose']]
    df = df.rename(columns={'ordercategorydescription': 'inputroute', 'amountuom': 'amountunit'})
    df.to_pickle(data_path + 'prep/{}.p'.format(fname.lower()))
    print()
    print('Done!')
    return df

def verify_output():
    print_header('Verify', char='=')
    count_dict = {
        'inputevents_mv': 2893707, # filtered; raw: 3585130
        'procedureevents_mv': 256001,
        'outputevents': 1549176,
        'datetimeevents': 2651118,
        'microbiologyevents': 174392, # specimen only; before remove duplicates: 283886
        'labevents': 10114036,
        'chartevents': 103868573,
    }
    for fname, n_rows in count_dict.items():
        print('-', fname)
        df = pd.read_pickle(data_path + 'prep/{}.p'.format(fname))
        assert n_rows == len(df), 'Expected {}, got {}'.format(n_rows, len(df))


def stack_attr_columns():
    """
    Input:
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    |  ID  |  t (or t_start + t_end)  |  ATTR_1  |  ATTR_2  |  ATTR_3  |  ...  |
    –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    
    Output:
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    |  ID  |  t (or t_start + t_end)  |  variable_name  |  variable_value  |
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    where the values of "variable_name" columns are: ATTR_1, ATTR_2, ATTR_3, ...
    """
    print_header('Formatting - Stack value columns', char='*')
    formatted_data = {}
    
    print("Stacking invariant table...")
    invariant_data = pd.read_csv(data_path + 'prep/invariant.csv', index_col='ID')
    # invariant_data['FIRST_WARDID'] = invariant_data['FIRST_WARDID'].apply(lambda x: 'WARDID=' + str(x))
    invariant_formatted = invariant_data.reset_index().melt(id_vars=['ID'], var_name='variable_name', value_name='variable_value')
    invariant_formatted['t'] = np.nan
    invariant_formatted = invariant_formatted[['ID', 't', 'variable_name', 'variable_value']]
    formatted_data['TIME_INVARIANT'] = invariant_formatted
    
    # save file for manual inspection only, not used in downstream processing
    invariant_formatted.to_csv(data_path + 'formatted/invariant_data.csv', index=False)

    print("Stacking event tables with attribute columns...")
    fnames = [
        'combined_lab',
        'combined_bio',
        'combined_dte',
        'combined_opt',
        'combined_chr',
        'combined_ipt',
        'combined_pcd',
    ]
    
    # columns that will be extracted from each table
    attribute_cols = {
        'combined_lab': ['value'],
        'combined_bio': [],
        'combined_dte': ['value'],
        'combined_opt': ['value'],
        'combined_chr': ['value'],
        'combined_ipt': ['inputroute', 'amount', 'rate', 'dose'],
        'combined_pcd': [],
    }
    
    for fname in tqdm(fnames):
        df = pd.read_pickle(data_path + 'prep/{}.p'.format(fname.lower()))
        df = df.rename(columns={'icustay_id': 'ID', 'itemid': 'variable_name'})

        should_melt = False
        t_cols = df.columns.intersection(['t', 't_start', 't_end']).tolist()

        if 't' in df.columns:
            id_cols = ['ID', 't', 'variable_name']
            # only changing the format if the value or other attribute cols exist
            if len(df.columns) > 3:
                should_melt = True
        elif 't_start' in df.columns and 't_end' in df.columns:
            id_cols = ['ID', 't_start', 't_end', 'variable_name']
            # only changing the format if the value or other attribute cols exist
            if len(df.columns) > 4:
                # print(len(df.columns))
                # print(df.columns)
                should_melt = True
        else:
            assert False

        df = df[id_cols + attribute_cols[fname]]

        # Extract each attribute column as a separate variable
        print(attribute_cols[fname])
        
        if len(attribute_cols[fname]) == 0:
            df_out = df
            df_out['variable_value'] = 1
        elif attribute_cols[fname] == ['value']:
            df_out = df.rename(columns={'value': 'variable_value'})
            df_out = df_out[['ID'] + t_cols + ['variable_name', 'variable_value']]
        elif should_melt:
            df_mask = df[id_cols].copy()
            df_mask['variable_value'] = 1
            df_attr = df.melt(id_vars=id_cols, var_name='attribute', value_name='variable_value')
            if fname == 'combined_ipt':
                df_attr['variable_name'] = df_attr['variable_name'].astype(str).str.cat(df_attr['attribute'].astype(str), sep='_')
            else:
                df_attr['variable_name'] = df_attr['variable_name'].str.cat(df_attr['attribute'], sep='_')
            df_attr = df_attr.drop(columns=['attribute'])
            df_out = pd.concat([df_mask, df_attr], sort=False, ignore_index=True)
            df_out = df_out.dropna(subset=['variable_value'])
        else:
            assert False

        formatted_data[fname] = df_out
    
    print('Done!')
    with open(data_path + 'formatted/all_data.stacked.p', 'wb') as f:
        pickle.dump(formatted_data, f)
    
    """
    `formatted_data` is a dictionary that maps each table name to a pd.DataFrame, 
    with either of the following two column formats:
    - for discrete time-stamps,
        –––––––––––––––––––––––––––––––––––––––––––––––––––
        |  ID  |  t  |  variable_name  |  variable_value  |
        –––––––––––––––––––––––––––––––––––––––––––––––––––
    - for continuous time-stamps,
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        |  ID  |  t_start  |  t_end  |  variable_name  |  variable_value  |
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    """
    return formatted_data


if __name__ == '__main__':
    main()

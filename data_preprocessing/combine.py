import pandas as pd
import os 

# Paths to the datasets
mimic_paths = {
    'mimic3': "physionet.org/files/mimiciii-demo/1.4/",
    'mimic4': "physionet.org/files/mimic-iv-demo/2.2/"
}

# Files to be loaded
data_files = {
    'lab': ("LABEVENTS.csv", "hosp/labevents.csv"),
    'bio': ("MICROBIOLOGYEVENTS.csv", "hosp/microbiologyevents.csv"),
    'dte': ("DATETIMEEVENTS.csv", "icu/datetimeevents.csv"),
    'opt': ("OUTPUTEVENTS.csv", "icu/outputevents.csv"),
    'chr': ("CHARTEVENTS.csv", "icu/chartevents.csv"),
    'ipt': ("INPUTEVENTS_MV.csv", "icu/inputevents.csv"),
    'pcd': ("PROCEDUREEVENTS_MV.csv", "icu/procedureevents.csv"),
    'icu': ("ICUSTAYS.csv", "icu/icustays.csv"),
    'itm': ("D_ITEMS.csv", "icu/d_items.csv"),
    'ltm': ("D_LABITEMS.csv", "hosp/d_labitems.csv"),
    'adm': ("ADMISSIONS.csv", "hosp/admissions.csv"),
    'pat': ("PATIENTS.csv", "hosp/patients.csv"),
    'dig': ("DIAGNOSES_ICD.csv", "hosp/diagnoses_icd.csv"),
    'drg': ("DRGCODES.csv", "hosp/drgcodes.csv")
}

def load_and_prepare_data(file_pair, path_dict, specific_columns=None):
    # Load the data
    path_3 = os.path.join(path_dict['mimic3'], file_pair[0])
    path_4 = os.path.join(path_dict['mimic4'], file_pair[1])

    df_3 = pd.read_csv(path_3)
    df_4 = pd.read_csv(path_4)
    
    # Convert column names to uppercase
    df_3.columns = [col.upper() for col in df_3.columns]
    df_4.columns = [col.upper() for col in df_4.columns]

    # Rename columns if necessary
    if 'STAY_ID' in df_4.columns:
        df_4.rename(columns={'STAY_ID': 'ICUSTAY_ID'}, inplace=True)
    if 'ICUSTAY_ID' in df_3.columns:
        df_3.rename(columns={'ICUSTAY_ID': 'ICUSTAY_ID'}, inplace=True)  # This seems redundant, adjust if there's a specific need

    # Handle specific columns for concatenation
    if specific_columns:
        columns_3 = specific_columns.get('mimic3', [])
        columns_4 = specific_columns.get('mimic4', [])

        # Make sure to include specific columns if they exist
        columns_3 = [col for col in columns_3 if col in df_3.columns]
        columns_4 = [col for col in columns_4 if col in df_4.columns]

        # Concatenate using the specific columns adjusted for each version
        combined_df = pd.concat([
            df_3[list(set(df_3.columns) & set(df_4.columns)) + columns_3],
            df_4[list(set(df_3.columns) & set(df_4.columns)) + columns_4]
        ])
    else:
        # Find shared columns and concatenate normally
        shared_columns = list(set(df_3.columns) & set(df_4.columns))
        combined_df = pd.concat([df_3[shared_columns], df_4[shared_columns]])

    return combined_df

# Process each dataset, add specific columns for 'pat'
specific_cols = {
    'pat': {'mimic3': ['DOB'], 'mimic4': ['ANCHOR_AGE']}
}

for key, file_pair in data_files.items():
    specific_columns = specific_cols.get(key)
    combined_df = load_and_prepare_data(file_pair, mimic_paths, specific_columns)
    combined_df.to_csv(f'combined_files/combined_{key}.csv', index=False)
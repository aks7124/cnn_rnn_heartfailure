import os
import codecs
import collections
import numpy as np
import pandas as pd
import string
import datetime as dt
import pickle

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'data')

def processing():
   
    diagICD = pd.read_csv(os.path.join(script_dir, 'data', 'DIAGNOSES_ICD.csv'))
    admissions = pd.read_csv(os.path.join(script_dir, 'data', 'ADMISSIONS.csv'))

    data = admissions[['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS']]

    # heart failure conditions
    h_failure_cond = ["HEART","CARDIAC","ARTERY","MYOCARDIAL"]
    # get hf data
    hf = data[(data['HOSPITAL_EXPIRE_FLAG'] == 1) & (data['DIAGNOSIS'].str.contains("|".join(h_failure_cond)))]
    data = pd.merge(data, hf, on='SUBJECT_ID', how='left')
    data = data.drop(columns=['DIAGNOSIS_y', 'HADM_ID_y'])
    data.columns = ['SUBJECT_ID', 'HADM_ID', 'HOSPITAL_EXPIRE_FLAG', 'DIAGNOSIS', 'H_FAILURE']

    # set 0 to non hf admissions
    data['H_FAILURE'].fillna(0, inplace=True)
    data['H_FAILURE'] = data['H_FAILURE'].astype(int)

    # get visits
    visits = pd.DataFrame(data['SUBJECT_ID'].value_counts())
    visits.reset_index(level=0, inplace=True)
    visits.columns = ['SUBJECT_ID', 'num_visits']
    vids = []
    for idx, row in visits.iterrows():
        vids.append(','.join([str(x) for x in range(row['num_visits'])] ))
    visits['VISITS'] = vids
    visits = visits.drop(columns=['num_visits'])    

    # drop diagnosis and hospital_expire_flag
    data = data.drop(columns=['HOSPITAL_EXPIRE_FLAG','DIAGNOSIS'])
    # remove all duplicates 
    data = data.drop_duplicates()
    # merge visits data
    data = data.merge(visits, on='SUBJECT_ID', how='left')

    # diagnosis codes
    diagICD = diagICD[diagICD['ICD9_CODE'].notna()]
    diagICD['ICD9_CODE'] = diagICD['ICD9_CODE']    
    seqs = diagICD.groupby(['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].apply(list).reset_index()

    # merge seqs
    data = pd.merge(data, seqs, on=['SUBJECT_ID', 'HADM_ID'])
    seqs = data[['SUBJECT_ID','ICD9_CODE']]
    seqs = seqs.groupby(['SUBJECT_ID'])['ICD9_CODE'].apply(list).reset_index()

    data = data.drop(columns=['HADM_ID', 'ICD9_CODE'])
    data = data.drop_duplicates()
    data = pd.merge(data, seqs, on=['SUBJECT_ID'])

    # update visits
    data['VISITS'] = data['VISITS'].apply(lambda x : [int(y) for y in x.split(',')])
        
    savePids(data)

    vids = data[['SUBJECT_ID', 'VISITS']]
    vids['SUBJECT_ID'].astype(int)
    saveVids(vids)

    hfs = data[['SUBJECT_ID', 'H_FAILURE']]    
    hfs.astype(int)
    saveHfs(hfs)

    seqs = data[['SUBJECT_ID', 'ICD9_CODE']]
    seqs['SUBJECT_ID'].astype(int)

    icdCodeMappings = getCodesMapping()

    saveSeqs(seqs, icdCodeMappings)

    saveICDCodes(icdCodeMappings)

    saveFull(data, icdCodeMappings)

    print('Done!')

def getCodesMapping():
    codesToSave = collections.defaultdict(str)
    with codecs.open(os.path.join(script_dir,'data', 'CMS32_DESC_LONG_DX.txt'), 'r', encoding='utf-8', errors='ignore') as in_file:
        for line in in_file:        
            codesToSave[line[:6].strip()] = (line[6:].strip()).translate(str.maketrans('','',string.punctuation))
    return codesToSave

    
def savePids(data):
    train, validate, test = np.split(data.sample(frac=1, random_state=1), [int(.6*len(data)), int(.8*len(data))])
    toSave = {'train' : train, 'validate' : validate, 'test' : test}
    for k, v in toSave.items():
        with open(os.path.join(os.path.join(data_dir, k), 'pids.pkl'), 'wb') as handle:
            pickle.dump(v['SUBJECT_ID'].tolist(), handle)

def saveVids(data):
    train, validate, test = np.split(data.sample(frac=1, random_state=1), [int(.6*len(data)), int(.8*len(data))])
    toSave = {'train' : train, 'validate' : validate, 'test' : test}
    for k, v in toSave.items():
        vidsToSave = v.set_index('SUBJECT_ID').to_dict()['VISITS']
        with open(os.path.join(os.path.join(data_dir, k), 'vids.pkl'), 'wb') as handle:
            pickle.dump(vidsToSave, handle)

def saveHfs(data):
    train, validate, test = np.split(data.sample(frac=1, random_state=1), [int(.6*len(data)), int(.8*len(data))])
    toSave = {'train' : train, 'validate' : validate, 'test' : test}
    for k, v in toSave.items():
        hfsToSave = v.set_index('SUBJECT_ID').to_dict()['H_FAILURE']
        with open(os.path.join(os.path.join(data_dir, k), 'hfs.pkl'), 'wb') as handle:
            pickle.dump(hfsToSave, handle)

def saveSeqs(data, icdCodeMappings):    
    codes_as_keys = list(icdCodeMappings.keys())
    
    train, validate, test = np.split(data.sample(frac=1, random_state=1), [int(.6*len(data)), int(.8*len(data))])
    toSave = {'train' : train, 'validate' : validate, 'test' : test}
    for type, v in toSave.items():
        seqsToSave = v.set_index('SUBJECT_ID').to_dict()['ICD9_CODE']
        seqsToSave = { k : [[ codes_as_keys.index(s) for s in s_l if s in codes_as_keys] for s_l in v] for k,v in seqsToSave.items() }
        with open(os.path.join(os.path.join(data_dir, type), 'seqs.pkl'), 'wb') as handle:
            pickle.dump(seqsToSave, handle)

def saveICDCodes(codesToSave):
    toSave = ['train', 'validate', 'test', 'full']
    for type in toSave:
        with open(os.path.join(os.path.join(data_dir, type), 'types.pkl'), 'wb') as handle:
            pickle.dump(list(codesToSave.keys()), handle)

        with open(os.path.join(os.path.join(data_dir, type), 'codes.pkl'), 'wb') as handle:
            pickle.dump(codesToSave, handle)

def saveFull(data, codes):
    # generate data files
    # pids    
    with open(os.path.join(os.path.join(data_dir, 'full'), 'pids.pkl'), 'wb') as handle:
        pickle.dump(data['SUBJECT_ID'].tolist(), handle)
    # vids
    vids = data[['SUBJECT_ID', 'VISITS']]
    vids['SUBJECT_ID'].astype(int)
    vidsToSave = vids.set_index('SUBJECT_ID').to_dict()['VISITS']
    with open(os.path.join(os.path.join(data_dir, 'full'), 'vids.pkl'), 'wb') as handle:
        pickle.dump(vidsToSave, handle)

    # hfs
    hfs = data[['SUBJECT_ID', 'H_FAILURE']]
    hfs.astype(int)
    hfsToSave = hfs.set_index('SUBJECT_ID').to_dict()['H_FAILURE']
    with open(os.path.join(os.path.join(data_dir, 'full'), 'hfs.pkl'), 'wb') as handle:
        pickle.dump(hfsToSave, handle)

    # seqs      
    codes_as_keys = list(codes.keys())
    seqs = data[['SUBJECT_ID', 'ICD9_CODE']]
    seqs['SUBJECT_ID'].astype(int)
    seqsToSave = seqs.set_index('SUBJECT_ID').to_dict()['ICD9_CODE']
    seqsToSave = { k : [[ codes_as_keys.index(s) for s in s_l if s in codes_as_keys] for s_l in v] for k,v in seqsToSave.items() }
    with open(os.path.join(os.path.join(data_dir, 'full'), 'seqs.pkl'), 'wb') as handle:
        pickle.dump(seqsToSave, handle)

def testData():
    def readDict(filePath, typ):
        with open(filePath, 'rb') as handle:
            data = pickle.load(handle)
            print('{} - {} is a dict with {} records.'.format(typ, os.path.basename(filePath), len(data)))

    for d in ['train', 'validate', 'test', 'full']:
        with open(os.path.join(data_dir, d,'pids.pkl'), 'rb') as handle:
            data = pickle.load(handle)
            print('{} - pids.pkl is a list with {} records.'.format(d, len(data)))
        readDict(os.path.join(data_dir, d, 'vids.pkl'), d)
        readDict(os.path.join(data_dir, d, 'pids.pkl'), d)
        readDict(os.path.join(data_dir, d, 'hfs.pkl'), d)
        readDict(os.path.join(data_dir, d, 'seqs.pkl'), d)
        readDict(os.path.join(data_dir, d, 'codes.pkl'), d)
        readDict(os.path.join(data_dir, d, 'types.pkl'), d)

processing()
testData()

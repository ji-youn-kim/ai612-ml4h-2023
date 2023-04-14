import argparse

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import collections
from collections import defaultdict
from copy import copy
import datetime
from datetime import timedelta,datetime
from transformers import AutoTokenizer

def get_parser():
    """
    Note:
        Do not add command-line arguments here when you submit the codes.
        Keep in mind that we will run your pre-processing code by this command:
        `python 00000000_preprocess.py ./train --dest ./output`
        which means that we might not be able to control the additional arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="root directory containing different ehr files to pre-process (usually, 'train/')"
    )
    parser.add_argument(
        "--dest",
        type=str,
        metavar="DIR",
        help="output directory"
    )
    return parser

def main(args):
    """
    TODO:
        Implement your feature preprocessing function here.
        Rename the file name with your student number.
    
    Note:
        This script should dump processed features to the --dest directory.
        Note that --dest directory will be an input to your dataset class (i.e., --data_path).
        You can dump any type of files such as json, cPickle, or whatever your dataset can handle.
    """

    root_dir = args.root
    dest_dir = args.dest

    preprocess_eicu(root_dir=root_dir, dest_dir=dest_dir)
    preprocess_mimic4(root_dir=root_dir, dest_dir=dest_dir)
########## eicu preprocess functions ##########

def preprocess_eicu(root_dir, dest_dir):

    def eicu_get_labels(label_path, patient_path):
        eicu_data = {}
        
        df = pd.read_csv(label_path)

        def map_labels(row):
            stay_id, labels = row
            eicu_data[stay_id] = {
                'inputs': [],
                'labels': np.array(eval(labels)),
                'pid': -1,
            }

        tqdm.pandas(desc='eicu | labels')
        df.progress_apply(map_labels, axis=1)

        pid2icustayids = defaultdict(set)

        def map_pid(row):
            stay_id, patient_id = row
            if stay_id in eicu_data:
                eicu_data[stay_id]['pid'] = patient_id
                pid2icustayids[patient_id].add(stay_id)

        df = pd.read_csv(patient_path, usecols=['patientunitstayid', 'uniquepid'])

        tqdm.pandas(desc='eicu | pid')
        df.progress_apply(map_pid, axis=1)

        
        return eicu_data, pid2icustayids

    def eicu_get_table(table_name, table_path, relevant_cols, offset_col, eicu_data):
        df = pd.read_csv(table_path, usecols=relevant_cols+['patientunitstayid', offset_col])

        # 2-pass vectorized
        for col_name in relevant_cols:
            df.insert(df.columns.get_loc(col_name), col_name+'_', col_name)
        df.insert(0, table_name, table_name)

        tqdm.pandas(desc=table_name)
        df['text'] = df.drop(['patientunitstayid', offset_col], axis=1).progress_apply(lambda x :' '.join(x.astype(str)), axis=1)

        def map_table(row, *args):
            text = row['text']
            offset = row[offset_col]
            args[0].append({
                'offset': offset,
                'text': text,
            })

        events = df.get(['patientunitstayid', offset_col, 'text']).groupby('patientunitstayid')
        for stay_id, group in tqdm(events, desc=table_name):
            if stay_id not in eicu_data: continue
            stay_id_event_list = eicu_data[stay_id]['inputs']
            group.apply(map_table, axis=1, args=(stay_id_event_list,))
        return eicu_data

    class Table():
        def __init__(self, table_name, table_path, relevant_cols, offset_col) -> None:
            self.table_name = table_name
            self.table_path = table_path
            self.relevant_cols = relevant_cols
            self.offset_col = offset_col
        def get(self):
            return self.table_name, self.table_path, self.relevant_cols, self.offset_col

    EICU_DIR = os.path.join(root_dir, 'eicu')
    LABELS_PATH = os.path.join(root_dir, 'labels', 'eicu_labels.csv')
    PATIENT_PATH = os.path.join(EICU_DIR, 'patient.csv')

    INTAKEOUTPUT_PATH = os.path.join(EICU_DIR, 'intakeOutput.csv')
    LAB_PATH = os.path.join(EICU_DIR, 'lab.csv')
    MEDICATION_PATH = os.path.join(EICU_DIR, 'medication.csv')
    # NURSECHARTING_PATH = os.path.join(EICU_DIR, 'nurseCharting.csv')

    tables_info = [
        Table(
            table_name='intakeOutput',
            table_path=INTAKEOUTPUT_PATH,
            relevant_cols=['intaketotal', 'outputtotal', 'dialysistotal', 'nettotal', 'cellpath', 'celllabel', 'cellvaluetext'],
            offset_col='intakeoutputentryoffset'
        ),
        Table(
            table_name='medication',
            table_path=MEDICATION_PATH,
            relevant_cols=['drugivadmixture', 'drugordercancelled', 'drugname', 'drughiclseqno', 'dosage', 'routeadmin', 'frequency', 'loadingdose', 'prn', 'gtc'],
            offset_col='drugorderoffset'
        ),
        Table(
            table_name='lab',
            table_path=LAB_PATH,
            relevant_cols=['labtypeid','labname','labresulttext','labmeasurenameinterface'],
            offset_col='labresultoffset'
        ),
    ]
    
    eicu_data, _ = eicu_get_labels(LABELS_PATH, PATIENT_PATH)

    for table in tables_info:
        eicu_data = eicu_get_table(*table.get(), eicu_data)

    small_eicu_data = copy(eicu_data)
    for icustay_id, icustay in eicu_data.items():
        if len(icustay['inputs']) == 0:
            del small_eicu_data[icustay_id]

    for icustay_id, icustay in tqdm(small_eicu_data.items(), desc='eicu | sort events'):
        pid = icustay['pid']
        events = icustay['inputs']
        sorted_events = sorted(events, key=lambda a: a['offset'])
        icustay['inputs'] = sorted_events

    total_num = len(small_eicu_data)
    train_num = total_num // 10 * 9
    val_num = total_num // 10 + total_num % 10
    assert total_num == train_num + val_num

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.truncation_side = 'left'

    # val_count = 0
    # val_icustay_ids = set()
    # for icustay_id, icustay in tqdm(small_eicu_data.items(), desc='eicu | split train/eval'):
    #     pid = icustay['pid']
    #     if len(pid2icustayids[pid]) == 1 and val_count < val_num:
    #         val_count += 1
    #         val_icustay_ids.add(icustay_id)

    for icustay_id, icustay in tqdm(small_eicu_data.items(), desc='eicu | save datafiles'):
        events = icustay['inputs']
        final_icustay = {
            'labels': icustay['labels']
        }
        final_events = []
        for event in events:
            final_events.append(event['text'])
        final_icustay['input'] = np.array(tokenizer(final_events, max_length=128, truncation=True, padding='max_length')['input_ids'])

        # if icustay_id in val_icustay_ids:
        #     final_icustay_path = os.path.join(dest_dir, 'val', f'eicu_{icustay_id}.pickle')
        # else:
        #     final_icustay_path = os.path.join(dest_dir, 'train', f'eicu_{icustay_id}.pickle')

        final_icustay_path = os.path.join(dest_dir, f'eicu_{icustay_id}.pickle')
        with open(final_icustay_path, 'wb') as f:
            pickle.dump(final_icustay, f)
def preprocess_mimic4(root_dir, dest_dir):
    
    MIMIC4_DIR = os.path.join(root_dir, 'mimiciv')
    LABELS_PATH = os.path.join(root_dir, 'labels', 'mimiciv_labels.csv')
    INPUTEVENT_PATH = os.path.join(MIMIC4_DIR, 'inputevents.csv')
    PRESCRIPTION_PATH = os.path.join(MIMIC4_DIR, 'prescriptions.csv')
    LABEVENTS_PATH = os.path.join(MIMIC4_DIR, 'labevents.csv')
    OUTPUTEVENTS_PATH = os.path.join(MIMIC4_DIR, 'outputevents.csv')
    D_LABITEMS_PAHT=os.path.join(MIMIC4_DIR, 'd_labitems.csv.gz')
    D_ITEMS_PATH = os.path.join(MIMIC4_DIR, 'd_items.csv.gz')
    
    inputevent = pd.DataFrame(pd.read_csv(INPUTEVENT_PATH))
    prescription = pd.DataFrame(pd.read_csv(PRESCRIPTION_PATH))
    labevents = pd.DataFrame(pd.read_csv(LABEVENTS_PATH))
    outputevent = pd.DataFrame(pd.read_csv(OUTPUTEVENTS_PATH))
    d_labitems = pd.DataFrame(pd.read_csv(D_LABITEMS_PAHT, compression='gzip', sep=','))
    d_itmes=pd.DataFrame(pd.read_csv(D_ITEMS_PATH, compression='gzip', sep=','))
    labels = pd.DataFrame(pd.read_csv(LABELS_PATH))
    
    def get_labitem_name(d_labitems):
        labitem_name =dict()
        for idx, row in tqdm(d_labitems.iterrows()):
            labitem_name[row['itemid']] = row['label']
        return labitem_name
    
    def get_item_name(d_items):
        item_name=dict()
        for idx, row in tqdm(d_itmes.iterrows()):
            item_name[row['itemid']]=row['label']
        return item_name
    
    def parsing(itemid):
        try:
            ans= labitem_name[itemid]
        except:
            ans =  None
        return ans
    
    def parsing2(itemid):
        try:
            ans= item_name[itemid]
        except:
            ans =  None
        return ans
    
    def stay_subject_mapping(inputevent):
        stay_hadm_dict = defaultdict(set)
        for idx,row in tqdm(inputevent.iterrows()):
            stay_hadm_dict[row['hadm_id']].add(row['stay_id'])
            
        return stay_hadm_dict
        
    def collect_inputevent(inputevent):
        input_dict = defaultdict(list)
        for idx, row in tqdm(inputevent.iterrows()):
            input_dict[row['stay_id']].append("inputevents")
            input_dict[row['stay_id']].append(row[['starttime','item_name', 'amount', 'amountuom', 'rate', 'rateuom', 'orderid',
            'linkorderid', 'ordercategoryname', 'secondaryordercategoryname',
            'ordercomponenttypedescription', 'ordercategorydescription',
            'patientweight', 'totalamount', 'totalamountuom', 'isopenbag',
            'continueinnextdept', 'statusdescription', 'originalamount',
            'originalrate']])
        
        return input_dict
        
    def collect_prescription(prescription, input_dict, stay_hadm_dict):
        prescription_input = input_dict.copy()
        for idx,row in tqdm(prescription.iterrows()):
            
            stayid = str(stay_hadm_dict[row['hadm_id']])
            prescription_input[stayid].append('prescriptions')
            prescription_input[stayid].append(row[['starttime','pharmacy_id', 'poe_id', 'poe_seq','drug_type', 'drug', 'formulary_drug_cd',
            'gsn', 'ndc', 'prod_strength', 'form_rx', 'dose_val_rx', 'dose_unit_rx',
            'form_val_disp', 'form_unit_disp', 'doses_per_24_hrs', 'route']])
            
        return prescription_input
    
    
    def collect_labinput(labevents, prescription_input, stay_hadm_dict):
        lab_input = prescription_input.copy()
        for idx, row in tqdm(labevents.iterrows()):
            
            stayid = str(stay_hadm_dict[row['hadm_id']])
            lab_input[stayid].append('labevents')
            lab_input[stayid].append(row[['charttime','item_name','value', 'valuenum', 'valueuom',
            'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments']])
        return lab_input
    
    
    def collect_outinput(outputevent, lab_input):
        out_input = lab_input.copy()
        for idx, row in tqdm(outputevent.iterrows()):
            out_input[row['stay_id']].append('outputevents')
            out_input[row['stay_id']].append(row[['charttime','item_name','value', 'valueuom']])
            
        return out_input
    
    def concat(out_input):
        result=[]
        stay_id_list=[]
        for stay_id,stay in out_input.items():
            #time_value=dict()
            time_value = defaultdict(list)
            for idx,event in enumerate(stay):
                if idx%2 ==0 : 
                    table_name = event
                else:
                    short_str=[]
                    for key,val in event.items():
                        if key=='starttime' or key=='charttime':
                            continue
                        else:
                            short_str.append(key.replace('\'',''))
                            short_str.append(val)
                        string = table_name+' '+' '.join(map(str,short_str))
                    time_value[event.values[0]].append(string)
            result.append(time_value)
            stay_id_list.append(stay_id)
        
        return result, stay_id_list
    
    def time_sorting(result,stay_id_list):
        final=dict()
        for idx,stay in enumerate(result):
            new_events = collections.OrderedDict(sorted(stay.items()))
            #string = ''.join(map(str,new_events.values()))
            string = list(new_events.values())

            final[stay_id_list[idx]]=string
        
        return final
    
    def tokenizing(dest_dir,final, tokenizer,labels):
        for stay_id,events in final.items():
            print_dict ={}
            tokens=[]
            
            for sentence in events:
                sentence = ' '.join(map(str,sentence))
                
                token = tokenizer(sentence)['input_ids']
                if len(token) >128:
                    token = token[:128]
                elif len(token) < 128:
                    token += [0]*(128-len(token))
                    
                tokens.append(token)
            try:
                label = labels[labels.stay_id==stay_id]['labels'].values[0].strip('[]')
                label = ''.join(map(str,label))
                label = np.fromstring(label, dtype=int, sep=',')
                
                print_dict["input"] = np.array(tokens)
                print_dict['label'] = label
                
                save_path = os.path.join(dest_dir, f'mimic4_{stay_id}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(print_dict,f)
                    
            except:
                continue
            
        return 
    
    labitem_name = get_labitem_name(d_labitems)
    item_name = get_item_name(d_itmes)
    labevents['item_name'] = None
    labevents['item_name'] = labevents.apply(lambda row: parsing(row['itemid']) ,axis=1 )
    outputevent['item_name'] = outputevent.apply(lambda row: parsing2(row['itemid']), axis=1)
    inputevent['item_name'] = inputevent.apply(lambda row: parsing2(row['itemid']), axis=1)
    
    ##collect all events by stay_id
    print("collecting")
    stay_hadm_dict = stay_subject_mapping(inputevent)
    input_event = collect_inputevent(inputevent)
    prescription_input = collect_prescription(prescription, input_event, stay_hadm_dict)
    lab_input = collect_labinput(labevents, prescription_input, stay_hadm_dict)
    out_input = collect_outinput(outputevent, lab_input)
    result, stay_id_list = concat(out_input)
    #sorting by time
    print("sorting")
    final = time_sorting(result,stay_id_list)
    #tokenize
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    print("saving")
    tokenizing(dest_dir,final, tokenizer,labels)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
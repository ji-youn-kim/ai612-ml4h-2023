import argparse
import os
import time 
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import collections
from collections import Counter
from datetime import datetime
from collections import defaultdict
from pandarallel import pandarallel
from copy import copy
from transformers import AutoTokenizer
from operator import itemgetter
import multiprocessing as mp

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

    parser.add_argument(
        "--sample_filtering",
        type=bool,
        default=True,
        help="indicator to prevent filtering from being applies to the test dataset."
    )

    parser.add_argument('--no_eicu', action='store_true')

    parser.add_argument('--no_mimiciii', action='store_true')

    parser.add_argument('--no_mimiciv', action='store_true')

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

    if not args.no_eicu:
        preprocess_eicu(root_dir=root_dir, dest_dir=dest_dir)

    if not args.no_mimiciii:
        preprocess_mimiciii(root_dir=root_dir, dest_dir=dest_dir, sample_filtering=args.sample_filtering)

    if not args.no_mimiciv:
        preprocess_mimiciv(root_dir=root_dir, dest_dir=dest_dir)


########## eicu preprocess functions ##########
def preprocess_eicu(root_dir, dest_dir):
    pandarallel.initialize(nb_workers=32, progress_bar=False)

    def eicu_get_labels(label_path, patient_path):
        eicu_data = {}
        
        df = pd.read_csv(label_path)

        def map_labels(row):
            stayid, labels = row
            eicu_data[stayid] = {
                'inputs': [],
                'labels': np.array(eval(labels)),
                'pid': -1,
            }

        tqdm.pandas(desc='eicu | get labels')
        df.progress_apply(map_labels, axis=1)

        pid2icustayids = defaultdict(set)

        def map_pid(row):
            stayid, patientid = row
            if stayid in eicu_data:
                eicu_data[stayid]['pid'] = patientid
                pid2icustayids[patientid].add(stayid)

        df = pd.read_csv(patient_path, usecols=['patientunitstayid', 'uniquepid'])

        tqdm.pandas(desc='eicu | pid')
        df.progress_apply(map_pid, axis=1)

        
        return eicu_data, pid2icustayids

    def eicu_get_table(table_name, table_path, relevant_cols, offset_col, eicu_data):
        df = pd.read_csv(table_path, usecols=relevant_cols+['patientunitstayid', offset_col], low_memory=False)

        # 2-pass vectorized
        for col_name in relevant_cols:
            df.insert(df.columns.get_loc(col_name), col_name+'_', col_name)
        df.insert(0, table_name, table_name)

        # tqdm.pandas(desc='eicu | join text: ' + table_name)
        print(f'eicu | join text ({table_name})')
        df['text'] = df.drop(['patientunitstayid', offset_col], axis=1).parallel_apply(lambda x :' '.join(x.astype(str)), axis=1)

        def map_table(row, *args):
            text = row['text']
            offset = row[offset_col]
            args[0].append({
                'offset': offset,
                'text': text,
            })

        events = df.get(['patientunitstayid', offset_col, 'text']).groupby('patientunitstayid')
        for stayid, group in tqdm(events, desc=f'eicu | group events ({table_name})'):
            if stayid not in eicu_data: continue
            stay_event_list = eicu_data[stayid]['inputs']
            group.apply(map_table, axis=1, args=(stay_event_list,))

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

    NURSEASSESSMENT_PATH = os.path.join(EICU_DIR, 'nurseAssessment.csv')
    NURSECARE_PATH = os.path.join(EICU_DIR, 'nurseCare.csv')
    NURSECHARTING_PATH = os.path.join(EICU_DIR, 'nurseCharting.csv')

    VITALAPERIODIC_PATH = os.path.join(EICU_DIR, 'vitalAperiodic.csv')
    VITALPERIODIC_PATH = os.path.join(EICU_DIR, 'vitalPeriodic.csv')

    tables_info = [
        Table(
            table_name='intakeOutput',
            table_path=INTAKEOUTPUT_PATH,
            relevant_cols=['intaketotal', 'outputtotal', 'dialysistotal', 'nettotal', 'cellpath', 'celllabel', 'cellvaluetext'],
            offset_col='intakeoutputentryoffset'
        ),
        Table(
            table_name='lab',
            table_path=LAB_PATH,
            relevant_cols=['labtypeid','labname','labresulttext','labmeasurenameinterface'],
            offset_col='labresultoffset'
        ),
        Table(
            table_name='medication',
            table_path=MEDICATION_PATH,
            relevant_cols=['drugivadmixture', 'drugordercancelled', 'drugname', 'drughiclseqno', 'dosage', 'routeadmin', 'frequency', 'loadingdose', 'prn', 'gtc'],
            offset_col='drugorderoffset'
        ),
        Table(
            table_name='nurseassessment',
            table_path=NURSEASSESSMENT_PATH,
            relevant_cols=['celllabel','cellattribute','cellattributevalue'],
            offset_col='nurseassessoffset'
        ),
        Table(
            table_name='nursecare',
            table_path=NURSECARE_PATH,
            relevant_cols=['cellattribute','cellattributevalue'],
            offset_col='nursecareoffset'
        ),
        Table(
            table_name='nursecharting',
            table_path=NURSECHARTING_PATH,
            relevant_cols=['nursingchartcelltypecat','nursingchartcelltypevallabel','nursingchartcelltypevalname','nursingchartvalue'],
            offset_col='nursingchartoffset'
        ),
        Table(
            table_name='vitalaperiodic',
            table_path=VITALAPERIODIC_PATH,
            relevant_cols=['noninvasivesystolic','noninvasivediastolic','noninvasivemean'],
            offset_col='observationoffset'
        ),
        Table(
            table_name='vitalperiodic',
            table_path=VITALPERIODIC_PATH,
            relevant_cols=['temperature','sao2','heartrate','respiration','cvp','etco2','systemicsystolic','systemicdiastolic','systemicmean','pasystolic','padiastolic','pamean','st1','st2','st3','icp'],
            offset_col='observationoffset'
        ),
    ]
    
    eicu_data, _ = eicu_get_labels(LABELS_PATH, PATIENT_PATH)

    for table in tables_info:
        eicu_data = eicu_get_table(*table.get(), eicu_data)

    small_eicu_data = copy(eicu_data)

    if args.sample_filtering:
        for stayid, icustay in eicu_data.items():
            if len(icustay['inputs']) == 0:
                del small_eicu_data[stayid]

    final_eicu_data = {}

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    for stayid, icustay in tqdm(small_eicu_data.items(), desc='eicu | sort and tokenize events'):
        # pid = icustay['pid']
        events = icustay['inputs']
        sorted_events = sorted(events, key=lambda a: a['offset'])
        final_icustay = {
            'labels': icustay['labels']
        }
        final_events = []
        for event in sorted_events[:256]:
            final_events.append(event['text'])
        final_icustay['input'] = np.array(tokenizer(final_events, max_length=128, truncation=True, padding='max_length')['input_ids'])

        final_eicu_data[stayid] = final_icustay

    # total_num = len(small_eicu_data)
    # train_num = total_num // 10 * 9
    # val_num = total_num // 10 + total_num % 10
    # assert total_num == train_num + val_num

    # val_count = 0
    # _icustayids = set()
    # for stayid, icustay in tqdm(small_eicu_data.items(), desc='eicu | split train/eval'):
    #     pid = icustay['pid']
    #     if len(pid2icustayids[pid]) == 1 and val_count < val_num:
    #         val_count += 1
    #         _icustayids.add(stayid)

    for stayid, final_icustay in tqdm(final_eicu_data.items(), desc='eicu | save datafiles'):
    # def map_save(icustay):
        # stayid, final_icustay = icustay

        final_icustay_path = os.path.join(dest_dir, f'eicu_{stayid}.pickle')
        with open(final_icustay_path, 'wb') as f:
            pickle.dump(final_icustay, f)

                  
########## mimiciii preprocess functions ##########
def preprocess_mimiciii(root_dir, dest_dir, sample_filtering):
    pandarallel.initialize(nb_workers=32, progress_bar=True)
    
    total_df = pd.DataFrame(columns=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'EVENT_SEQ'])
    hadm_icu_dict = {}

    # tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # identification files
    data_dir_path = os.path.join(root_dir, "mimiciii/")
    d_items = pd.read_csv(os.path.join(data_dir_path, "D_ITEMS.csv"))
    d_labitems = pd.read_csv(os.path.join(data_dir_path, "D_LABITEMS.csv"))

    d_items_dict = dict(zip(d_items["ITEMID"], d_items["LABEL"]))
    d_labitems_dict = dict(zip(d_labitems["ITEMID"], d_labitems["LABEL"]))

    col_names_per_table = {
        "labevents" : ["ITEMID", "VALUE", "VALUENUM", "VALUEUOM", "FLAG"],
        "prescriptions" : ["DRUG_TYPE", "DRUG" ,"DRUG_NAME_POE", "DRUG_NAME_GENERIC","FORMULARY_DRUG_CD","GSN",\
                        "NDC","PROD_STRENGTH","DOSE_VAL_RX","DOSE_UNIT_RX","FORM_VAL_DISP","FORM_UNIT_DISP","ROUTE"],
        "inputevents_cv" : ["ITEMID","AMOUNT","AMOUNTUOM","RATE","RATEUOM","STOPPED","NEWBOTTLE","ORIGINALAMOUNT",\
                            "ORIGINALAMOUNTUOM","ORIGINALROUTE","ORIGINALRATE","ORIGINALRATEUOM","ORIGINALSITE"],
        "inputevents_mv" : ["ITEMID", "AMOUNT","AMOUNTUOM","RATE","RATEUOM", "ORDERCATEGORYNAME",\
                    "SECONDARYORDERCATEGORYNAME","ORDERCOMPONENTTYPEDESCRIPTION","ORDERCATEGORYDESCRIPTION","PATIENTWEIGHT",\
                    "TOTALAMOUNT","TOTALAMOUNTUOM","ISOPENBAG","CONTINUEINNEXTDEPT","CANCELREASON","STATUSDESCRIPTION", \
                    "COMMENTS_CANCELEDBY", "ORIGINALAMOUNT","ORIGINALRATE"],
        "outputevents" : ["ITEMID","VALUE","VALUEUOM","STOPPED","NEWBOTTLE","ISERROR"],
        "chartevents" : ["ITEMID", "VALUE", "VALUENUM", "VALUEUOM"]
    }

    
    mimiciii_top200_items = pickle.load(open("./mimiciii_top200.pkl", "rb"))  # ["mimiciii"]

    """Functions"""
    def apply_charttime(row):
        table_name = row["TABLE_NAME"]
        if table_name == "prescriptions":
            charttime = datetime.strptime(
                row["STARTDATE"]+" 00:00:00", "%Y-%m-%d %H:%M:%S")
        elif table_name == "inputevents_mv":
            charttime = datetime.strptime(
                row["STARTTIME"], "%Y-%m-%d %H:%M:%S")
        else:
            charttime = datetime.strptime(
                row["CHARTTIME"], "%Y-%m-%d %H:%M:%S")
        return charttime
    
    def apply_event_seq(row):
        table_name = row["TABLE_NAME"].lower()
        col_names = col_names_per_table[table_name]

        event_seq = f"{table_name} "
        
        for col_name in col_names:
            c_val = row[col_name]
            c_name = col_name.lower()
        
            if c_name=="itemid":
                c_val = d_labitems_dict[c_val] if table_name == "labevents" else d_items_dict[c_val]
            elif c_name=="valuenum":
                c_val = str(c_val)

            if not pd.isnull(c_val):
                event_seq += f"{c_name} {c_val} "

        return  event_seq.strip() 


    def chunk_parallel(table_name, table_path, total_df):
        chunksize = 10**7
        print(table_name)
        i =0
        for cnt, chunk in enumerate(pd.read_csv(table_path, chunksize=chunksize)):
            print("chunk_num: ", cnt+1)

            t0 = time.time()
            
            if table_name=="ICUSTAYS":
                for idx, row in tqdm(chunk.iterrows(), desc=table_name.lower()):
                    hadm_id, icustay_id =  row["HADM_ID"], row["ICUSTAY_ID"]
                    
                    hadm_icu_dict[hadm_id] = icustay_id
                    
            else:
                if table_name == "CHARTEVENTS":
                    chunk = chunk[chunk["ITEMID"].isin(mimiciii_top200_items)]
                
                chunk["TABLE_NAME"] = table_name.lower()
                chunk["CHARTTIME"] = chunk.parallel_apply(apply_charttime, axis=1, )
                #if 'STARTTIME' in chunk.columns:
                #    chunk.drop(['STARTTIME'], axis=1, inplace=True)
                #print(chunk.columns)
                if "ICUSTAY_ID" not in chunk.columns:
                    chunk["ICUSTAY_ID"] = chunk["HADM_ID"].parallel_apply(lambda x : hadm_icu_dict[x])
                
                chunk["EVENT_SEQ"] = chunk.parallel_apply(apply_event_seq, axis=1)
                
                chunk.sort_values(
                    by=['HADM_ID', "ICUSTAY_ID", "CHARTTIME"], inplace=True)

                preprocessed_chunk = chunk[[
                    "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "EVENT_SEQ"]].dropna(axis=0)
                
                preprocessed_chunk = preprocessed_chunk.sort_values(by="CHARTTIME")

                total_df = pd.concat([total_df, preprocessed_chunk])
                
            print(
                f"{table_name}_processes run time {time.time() - t0:f} seconds.")
        return total_df
            
            
    """Main Code"""
    # preprocess each table
    table_names = ["ICUSTAYS", "LABEVENTS", "PRESCRIPTIONS", "INPUTEVENTS_CV", "INPUTEVENTS_MV", "OUTPUTEVENTS", "CHARTEVENTS"]
    for table_name in table_names:
        table_path = os.path.join(data_dir_path, f"{table_name}.csv")
        total_df = chunk_parallel(table_name, table_path, total_df)

    # make {inputs : [e, e, e, ...], label : [0, 1, 2, 0, ..., -1, 1, 0]}
    mimiciii_labels = pd.read_csv(os.path.join(root_dir, "labels/mimiciii_labels.csv"))
    mimiciii_labels_dict = dict(zip(mimiciii_labels["ICUSTAY_ID"], mimiciii_labels["labels"]))
   
    total_df = total_df.drop(["HADM_ID"], axis=1)
    
    curr_icustays = []
    for icustay_id, group in tqdm(total_df.groupby("ICUSTAY_ID")):
        curr_icustays.append(icustay_id)
        tokenized_events = []
        #print(group.iloc[-100:, :])
        
        if len(group) > 256:
            group = group.iloc[:256, :]
        
        group = group.sort_values(by=["CHARTTIME"])

        #print(group["EVENT_SEQ"].values.shape)
        for event_seq in group["EVENT_SEQ"].values:
            tokenized = tokenizer.encode(event_seq)
            if len(tokenized) < 128:
                tokenized += [0] * (128-len(tokenized))
            elif len(tokenized) > 128:
                tokenized = tokenized[:128]
            tokenized_events.append(tokenized)

        if len(mimiciii_labels[mimiciii_labels["ICUSTAY_ID"] == icustay_id]) > 0:
            labels = mimiciii_labels_dict[icustay_id]
            labels = np.array(eval(labels))
            
            tokenized_events = np.array(tokenized_events)
            icu_stay_dict = {"input": tokenized_events, "label": labels}
            with open(file=os.path.join(dest_dir, f'mimiciii_{icustay_id}.pickle'), mode='wb') as f:
                pickle.dump(icu_stay_dict, f)

    if not sample_filtering:
        left_samples = list(set(mimiciii_labels_dict.keys()) - set(curr_icustays))
        print("number of samples with no events: ", len(left_samples))
        for icustay_id in tqdm(left_samples, desc="samples with no events"):
            labels = np.array(eval(mimiciii_labels_dict[icustay_id]))
            icu_stay_dict = {"input": np.array([]), "label": labels}
            with open(file=os.path.join(dest_dir, f'mimiciii_{icustay_id}.pickle'), mode='wb') as f:
                pickle.dump(icu_stay_dict, f)
    return 

########## mimiciv preprocess functions ##########
def preprocess_mimiciv(root_dir, dest_dir):
    
    MIMIC4_DIR = os.path.join(root_dir, 'mimiciv')
    LABELS_PATH = os.path.join(root_dir, 'labels', 'mimiciv_labels.csv')
    INPUTEVENT_PATH = os.path.join(MIMIC4_DIR, 'inputevents.csv')
    PRESCRIPTION_PATH = os.path.join(MIMIC4_DIR, 'prescriptions.csv')
    LABEVENTS_PATH = os.path.join(MIMIC4_DIR, 'labevents.csv')
    OUTPUTEVENTS_PATH = os.path.join(MIMIC4_DIR, 'outputevents.csv')
    D_LABITEMS_PAHT=os.path.join(MIMIC4_DIR, 'd_labitems.csv.gz')
    D_ITEMS_PATH = os.path.join(MIMIC4_DIR, 'd_items.csv.gz')
    CHARTEVENTS_PATH = os.path.join(MIMIC4_DIR, 'chartevents.csv')
    
    inputevent = pd.DataFrame(pd.read_csv(INPUTEVENT_PATH))
    prescription = pd.DataFrame(pd.read_csv(PRESCRIPTION_PATH))
    labevents = pd.DataFrame(pd.read_csv(LABEVENTS_PATH))
    outputevent = pd.DataFrame(pd.read_csv(OUTPUTEVENTS_PATH))
    d_labitems = pd.DataFrame(pd.read_csv(D_LABITEMS_PAHT, compression='gzip', sep=','))
    d_itmes=pd.DataFrame(pd.read_csv(D_ITEMS_PATH, compression='gzip', sep=','))
    labels = pd.DataFrame(pd.read_csv(LABELS_PATH))
    chartevents = list(pd.read_csv(CHARTEVENTS_PATH, chunksize=10000))
    
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
    
    def multi_processing(chartevents_list,top100_itemid):
        global delete_rows
        def delete_rows(df, top100_itemid):
            df = pd.DataFrame(df)
            del_list=[]
            for idx in df.index:
                if df.loc[idx, 'itemid'] not in top100_itemid:
                    del_list.append(idx)
            df.drop(del_list, axis=0, inplace = True)   
            return df 
  
        pool = mp.Pool(15) # use 4 processes

        funclist = []
        for df in tqdm(chartevents_list):
            f = pool.apply_async(delete_rows,(df, top100_itemid))
            funclist.append(f)
  
        chart_result=pd.DataFrame(columns=['subject_id', 'hadm_id', 'stay_id', 'charttime', 'storetime', 'itemid',
       'value', 'valuenum', 'valueuom', 'warning'])
        
        for f in tqdm(funclist):
            chart_result = pd.concat([chart_result, f.get()])
        chart_result.to_csv(os.path.join(MIMIC4_DIR, 'top150_chartevents.csv'))
    
        return chart_result
    
    def create_chart_dict(chartevents_list, lack_stayid):
        global collect_chartevnts
        def collect_chartevnts(df, lack_stayid):
            chartevent_group = defaultdict(dict)
            for idx,row in df.iterrows():
                    if row['stay_id'] in lack_stayid:
                            chartevent_group[row['stay_id']][row['charttime']]=row[['item_name','value','valuenum','valueuom','warning' ]]
                    else:
                            continue

            chart_dict=defaultdict(dict)
            for stayid, events in chartevent_group.items():
                    tmp=dict()
                    #sorted_chartevent=dict()
                    #sorted_events = collections.OrderedDict(sorted(events.items() ))
                    for time, explain in events.items() :
                            tmp[time] = explain
                    chart_dict[stayid] = tmp
        
            return chart_dict
        
        pool = mp.Pool(15) 
        
        funclist = []
        for df in tqdm(chartevents_list):
            f = pool.apply_async(collect_chartevnts,(df, lack_stayid))
            funclist.append(f)
            
        chart_dict=defaultdict(dict)
        for f in tqdm(funclist):
            chunk_dict = f.get()
            for stayid,events in chunk_dict.items():
                chart_dict.update({stayid : events})
        
        return chart_dict
    
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
    
    ##adding chartevents
    print("adding chartevent")
    item_dict= Counter()
    for chunk in tqdm(chartevents):
        chunkdf = chunk.itemid
        counter = Counter(chunkdf)
        item_dict+=counter
    sorted_C = sorted(item_dict.items(), key=itemgetter(1), reverse=True)
    top100_itemid=[]
    for i in range(150):
        top100_itemid.append(sorted_C[i][0])
        
    print("multiprocessing")
    chart_result = multi_processing(chartevents, top100_itemid)
    
    print("saving temoporary chartevents")
    preprocessed_charts = pd.read_csv(os.path.join(MIMIC4_DIR, 'top150_chartevents.csv'))
    preprocessed_chart_list = list(pd.read_csv(os.path.join(MIMIC4_DIR, 'top150_chartevents.csv'),chunksize=10000))
    preprocessed_charts['item_name'] = preprocessed_charts.apply(lambda row: parsing2(row['itemid']), axis=1)
    preprocessed_charts.to_csv(os.path.join(MIMIC4_DIR,'tmp_preprocess_charts.csv'))
    chartevents_list = list(pd.read_csv(os.path.join(MIMIC4_DIR,'tmp_preprocess_charts.csv'), chunksize=10000))
    
    lack_stayid=[]
    for key,val in out_input.items():
        if len(val)<256*2:
            lack_stayid.append(key)
        else:
            out_input[key] = val[:256*2] 
    
    print("creating chart dictionary time:events")
    chart_dict = create_chart_dict(chartevents_list, lack_stayid)
            
    sorted_chart = defaultdict(list)
    for stayid,events in tqdm(chart_dict.items()):
        sorted_events = collections.OrderedDict(sorted(events.items() ))

        for time,events in sorted_events.items():
            short_str=[]
            for key,val in events.items():
                short_str.append(key.replace('\'',''))
                short_str.append(val)
            string = 'chartevent'+' '+' '.join(map(str,short_str))
            sorted_chart[stayid].append(time)
            sorted_chart[stayid].append(string)
    
    print("concatting to make 256")
    tmp_result = result.copy()
    for idx,stayid in enumerate(stay_id_list):
        if stayid in lack_stayid:
            needed = 256-len(result[idx])
            for i in range(needed):
                try : 
                    time = sorted_chart[stayid][2*i]
                    string = sorted_chart[stayid][2*i+1]
                    tmp_result[idx].update({time: [string]})
                except:
                    break
                
    #sorting by time
    print("sorting")
    final = time_sorting(tmp_result,stay_id_list)
    #tokenize
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    print("saving")
    tokenizing(dest_dir,final, tokenizer,labels)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

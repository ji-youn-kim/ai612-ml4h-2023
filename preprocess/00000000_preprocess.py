import argparse

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
from copy import copy
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
        "root",
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

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
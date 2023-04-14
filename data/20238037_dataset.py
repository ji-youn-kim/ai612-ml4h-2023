from . import BaseDataset, register_dataset

import os
import torch
import numpy as np
import pandas as pd

@register_dataset("20238037_dataset")
class MyDataset20238037(BaseDataset):
    """
    TODO:
        create your own dataset here.
        Rename the class name and the file name with your student number
    
    Example:
    - 20218078_dataset.py
        @register_dataset("20218078_dataset")
        class MyDataset20218078(BaseDataset):
            (...)
    """

    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        max_event_size=256,
        max_token_size=128,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_event_size = max_event_size
        self.max_token_size = max_token_size

        self.files = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]
    
    def __getitem__(self, index):

        item = pd.read_pickle(os.path.join(self.data_path, self.files[index]))
        item['input'] = np.array([i for i in item['input'] if np.any(i)])
        # item['fname'] = self.files[index]
        if 'labels' in item.keys():
            item['label'] = item['labels']
            del item['labels']
    
        return item
    
    def __len__(self):
        
        return len(self.files)

    def collator(self, samples):
        
        samples = [s for s in samples if s['input'] is not None]
    
        if len(samples) == 0:
            return {}
        input = dict()
        input['input'] = [s['input'] for s in samples]
        target = dict()
        target['label'] = [s['label'] for s in samples]

        seq_sizes, token_sizes = [], []
        for idx, s in enumerate(input['input']):
            seq_sizes.append(np.shape(s)[0])
            token_sizes.append(np.shape(s)[1])

        target_event_size = min(max(seq_sizes), self.max_event_size)

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros(
                (len(input['input']), target_event_size, self.max_token_size)
            )

        for i, (seq_size, token_size) in enumerate(zip(seq_sizes, token_sizes)):
            t_diff = token_size - self.max_token_size
            s_diff = seq_size - target_event_size
            for k in input.keys():
                if k == 'input':
                    prefix = 101

                if t_diff < 0:
                    t_padding = np.zeros((seq_size, -t_diff))
                    input[k][i] = np.concatenate(
                        [input[k][i], t_padding], axis=1
                    )
                elif t_diff > 0:
                    input[k][i] = input[k][i][:, :self.max_token_size]

                if s_diff == 0:
                    collated_input[k][i] = torch.from_numpy(input[k][i])
                elif s_diff < 0:
                    padding = np.zeros((-s_diff, self.max_token_size - 1))
                    padding = np.concatenate(
                        [np.full((-s_diff, 1), fill_value=prefix), padding], axis=1
                    )
                    collated_input[k][i] = torch.from_numpy(
                        np.concatenate(
                        [input[k][i], padding], axis=0
                        )
                    )
                else:
                    collated_input[k][i] = torch.from_numpy(input[k][i][:target_event_size])
            
        collated_input['input'] = collated_input['input'].type(torch.int64)
        collated_input['label'] = torch.LongTensor(np.array([s['label'] for s in samples]))
        # collated_input['fname'] = np.array([s['fname'] for s in samples])

        # shape: {'input': (B, E, S), 'label': (B, 28)}
        return collated_input 
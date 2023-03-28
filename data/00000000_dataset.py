from . import BaseDataset, register_dataset

@register_dataset("00000000_dataset")
class MyDataset00000000(BaseDataset):
    """
    TODO: create your own dataset here.
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
        # ...,
        **kwargs,
    ):
        super().__init__()
        ...
    
    def __getitem__(self, index):
        """
        Note:
            You must return a dictionary here or in collator.
            Example:
                def __getitem__(self, index):
                    (...)
                    return {"data": data, "label": label}
        """
        ...
    
    def __len__(self):
        ...

    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.
        
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        
        Note:
            You can use it to make your batch on your own such as outputting padding mask together.
            Otherwise, you don't need to implement this method.
        """
        raise NotImplementedError
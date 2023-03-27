from . import BaseDataset, register_dataset

@register_dataset("00000000_dataset")
class MyDataset00000000(BaseDataset):
    """
    TODO: create your own dataset here and replace the numbers in the class name with your
        student number
    
    Example:
        @register_dataset("20218078_dataset")
        class MyDataset20218078(BaseDataset):
            (...)
    """

    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        ...,
        **kwargs,
    ):
        super().__init__()
        ...
    
    def __getitem__(self, index):
        ...
    
    def __len__(self):
        ...

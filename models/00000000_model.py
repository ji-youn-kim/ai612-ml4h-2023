from typing import Dict

from . import BaseModel, register_model

@register_model("00000000_model")
class MyModel00000000(BaseModel):
    """
    TODO:
        create your own model here to handle heterogeneous EHR formats.
        Rename the class name and the file name with your student number.
    
    Example:
    - 20218078_model.py
        @register_model("20218078_model")
        class MyModel20218078(BaseModel):
            (...)
    """
    
    def __init__(
        self,
        # ...,
        **kwargs,
    ):
        super().__init__()
        ...
    
    def get_logits(cls, net_output):
        """get logits from the net's output.
        
        Note:
            Assure that get_logits(...) should return the logits in the shape of (batch, 52)
        """
        ...
    
    def get_targets(self, sample):
        """get targets from the sample
        
        Note:
            Assure that get_targets(...) should return the ground truth labels
                in the shape of (batch, 28)
        """
        ...

    def forward(
        self,
        # ...,
        **kwargs
    ):
        """
        Note:
            the key should be corresponded with the output dictionary of the dataset you implemented.
        
        Example:
            class MyDataset(...):
                ...
                def __getitem__(self, index):
                    (...)
                    return {"data_key": data, "label": label}
            
            class MyModel(...):
                ...
                def forward(self, data_key, **kwargs):
                    (...)
        """
        ...
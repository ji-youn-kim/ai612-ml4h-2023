from . import BaseModel, register_model

@register_model("00000000_model")
class MyModel00000000(BaseModel):
    """
    TODO: create your own model here to handle heterogeneous EHR formats.
    Replace the numbers in the class name with your student number
    
    Example:
        @register_model("20218078_model")
        class MyModel20218078(BaseModel):
            (...)
    """
    
    def __init__(
        self,
        ...,
        **kwargs,
    ):
        super().__init__()
        ...
    
    def get_logits(cls, net_output):
        """get logits from the net's output."""
        # NOTE: assure that get_logits(...) should return the logits in the shape of (batch, 29, dim)
        ...
    
    def get_targets(self, sample):
        """get targets from the sample"""
        # NOTE: assure that get_targets(...) shuold return the ground truth labels
        #   in the shape of (batch, 29)
        ...

    def forward(
        self,
        ...
    ):
        ...
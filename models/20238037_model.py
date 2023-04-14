from typing import Dict
import math

import torch
import torch.nn as nn

from . import BaseModel, register_model

@register_model("20238037_model")
class MyModel20238037(BaseModel):
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
        **kwargs,
    ):
        super().__init__()

        self.input_emb = TokenEmbedding(**kwargs)
        self.event_encoder = TransformerEventEncoder(**kwargs)
        self.event_aggregator = TransformerEventAggregator(**kwargs)
        self.predict = PredictOutput(**kwargs)
    
    def get_logits(cls, net_output): # 
        """get logits from the net's output.
        
        Note:
            Assure that get_logits(...) should return the logits in the shape of (batch, 52)
        """
        return net_output
    
    def get_targets(self, sample):
        """get targets from the sample
        
        Note:
            Assure that get_targets(...) should return the ground truth labels
                in the shape of (batch, 28)
        """
        return sample["label"]

    def forward(
        self,
        input, # (B, E, S)
        **kwargs
    ):
        # 1. Encode Event (B, E, S) -> (B, E, emb_in)
        # 2. Aggregate Event (B, E, emb_in) -> (B, emb_out)
        emb_event = self.input_emb(input) # (B, E, S) -> (B*E, S, Emb_in)
        # if emb_event.isnan().any():
        #     breakpoint()
        encode_event = self.event_encoder(emb_event, input) # (B*E, S, Emb_in) -> (B, E, Emb_out)
        # if encode_event.isnan().any():
        #     breakpoint()
        aggregate_event = self.event_aggregator(encode_event, input) # (B, E, Emb_out) -> (B, E, Emb_out)
        # if aggregate_event.isnan().any():
        #     breakpoint()
        predict = self.predict(aggregate_event, input)
        # if predict.isnan().any():
        #     breakpoint()

        return predict


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        emb_in=128,
        dropout_in=0.2,
        max_token_size=128,
        **kwargs,
    ):
        super().__init__()

        self.emb_in = emb_in

        self.tokenizer_size = 28996
        self.input_emb = nn.Embedding(self.tokenizer_size, self.emb_in, padding_idx=0)
        self.pos_encoder = PositionalEncoding(
            emb_in, dropout_in, max_token_size
        )
        self.layer_norm = nn.LayerNorm(emb_in, eps=1e-12)

    def forward(
        self,
        input,
        **kwargs
    ):
        B, E, S = input.size()
        emb_event = self.input_emb(input) # (B, E, S) -> (B, E, S, Emb_in)
        emb_event = emb_event.view(B*E, S, self.emb_in) # (B, E, S, Emb_in) -> (B*E, S, Emb_in)
        
        emb_event = self.layer_norm(self.pos_encoder(emb_event))

        return emb_event
    

class TransformerEventEncoder(nn.Module):
    def __init__(
        self,
        emb_in=128,
        emb_out=128,
        nhead_in=8,
        dropout_in=0.2,
        num_layers_in=2,
        **kwargs,
    ):
        super().__init__()
        self.emb_in, self.emb_out, self.nhead_in, self.dropout_in, self.num_layers_in = emb_in, emb_out, nhead_in, dropout_in, num_layers_in
        encoderlayer = nn.TransformerEncoderLayer(
            d_model=self.emb_in,
            nhead=self.nhead_in,
            dim_feedforward=self.emb_in*4,
            dropout=self.dropout_in,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoderlayer, self.num_layers_in
        )
        self.proj_output = nn.Linear(self.emb_in, self.emb_out)

    def forward(
        self,
        emb_event, # (B*E, S, Emb_in)
        input,
        **kwargs
    ):
        
        B, E, S = input.size()
        pad_mask = (
            input.view(B * E, -1).eq(0).to(emb_event.device)
        )
        # encode_event = self.encoder(emb_event, src_key_padding_mask=pad_mask) # (B*E, S, Emb_in)
        encode_event = self.encoder(emb_event)
        # apply mean pooling
        encode_event[pad_mask] = 0
        # breakpoint()
        pooled_event = torch.div(encode_event.sum(dim=1), (encode_event!=0).sum(dim=1)) # (B*E, Emb_in)

        proj = self.proj_output(pooled_event) # (B*E, Emb_out)
        proj = proj.view(B, E, self.emb_out) # (B, E, Emb_out)

        return proj # (B, E, Emb_out)
    

class TransformerEventAggregator(nn.Module):
    def __init__(
        self,
        emb_out=128,
        nhead_out=8,
        dropout_out=0.2,
        num_layers_out=2,
        max_event_size=256,
        **kwargs,
    ):
        super().__init__()
        self.emb_out, self.nhead_out, self.dropout_out, self.num_layers_out, self.max_event_size = emb_out, nhead_out, dropout_out, num_layers_out, max_event_size
        self.pos_encoder = PositionalEncoding(
            self.emb_out, self.dropout_out, self.max_event_size
        )
        self.layer_norm = nn.LayerNorm(self.emb_out, eps=1e-12)

        encoderlayer = nn.TransformerEncoderLayer(
            d_model=self.emb_out,
            nhead=self.nhead_out,
            dim_feedforward=self.emb_out*4,
            dropout=self.dropout_out,
            batch_first=True
        )
        self.aggregator = nn.TransformerEncoder(
            encoderlayer, self.num_layers_out
        )

    def forward(
        self,
        encode_event,
        input,
        **kwargs
    ):

        encode_event = self.layer_norm(self.pos_encoder(encode_event))
        
        pad_mask = input[:, :, 1].eq(0).to(encode_event.device)
        aggregate_event = self.aggregator( # (B, E, Emb_out)
            encode_event, mask=None, src_key_padding_mask=pad_mask
        )

        return aggregate_event


class PredictOutput(nn.Module):
    def __init__(
        self,
        emb_out=128,
        tasks=["mort_short", "mort_long", "readm", "dx", "los_short", "los_long", "fi_ac", "im_disch", "creat", "bili", "plate", "wbc"],
        **kwargs,
    ):
        super().__init__()
        self.emb_out, self.tasks = emb_out, tasks
        self.final_proj = nn.ModuleDict()
        classes = {
            "mort_short": 1, "mort_long": 1, "readm": 1, "dx": 17, "los_short": 1, "los_long": 1, 
            "fi_ac": 6, "im_disch": 6, "creat": 5, "bili": 5, "plate": 5, "wbc": 3
        }
        for task in self.tasks:
            self.final_proj[task] = nn.Linear(
                self.emb_out, classes[task]
            )
    
    def forward(
        self,
        aggregate_event, # (B, E, Emb_out)
        input,
        **kwargs
    ):
        
        B, E, S = input.size()
        
        # apply mean pooling
        pad_mask = (input[:, :, 1] == 0)
        aggregate_event[pad_mask] = 0
        aggregate_event = torch.div(aggregate_event.sum(dim=1), (aggregate_event!=0).sum(dim=1)) # (B, Emb_out)

        preds = []
        for _, layer in self.final_proj.items():
            preds.append(layer(aggregate_event)) 

        output = torch.cat(preds, 1)

        return output # (B, 52)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, sequence_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(sequence_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000) / dim)
        )
        pe = torch.zeros(1, sequence_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        
        x = self.dropout(x + self.pe[:, :x.size(1)])

        return x
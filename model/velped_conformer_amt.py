# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchaudio.models import Conformer

class AMTConformer(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,          
        encoder_dim: int = 384,
        num_heads: int = 6,         
        ffn_dim: int | None = None, # if None then 4 * encoder_dim
        num_layers: int = 6,
        dropout: float = 0.1,
        n_pitches: int = 88,
        conv_kernel: int = 31,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * encoder_dim
        assert encoder_dim % num_heads == 0, "encoder_dim must be divisible by num_heads"

        # Project mel features to model dimension
        self.in_proj = nn.Linear(n_mels, encoder_dim)

        # Conformer encoder expecting (B, T, encoder_dim)
        self.encoder = Conformer(
            input_dim=encoder_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
        )

        self.onset_head = nn.Linear(encoder_dim, n_pitches)
        self.frame_head = nn.Linear(encoder_dim, n_pitches)
        self.offset_head = nn.Linear(encoder_dim, n_pitches)
        self.vel_head = nn.Linear(encoder_dim, n_pitches)

        self.ped64_head = nn.Linear(encoder_dim, 1)   # sustain
        self.ped67_head = nn.Linear(encoder_dim, 1)   # una corda

        # Onset frame conditioning!!
        self.frame_cond  = nn.Linear(n_pitches, encoder_dim)

    def forward(self, x: torch.Tensor,  teach_on: torch.Tensor | None = None): 
        # (B, n_mels, T) to (B, T, encoder_dim) from some error
        x = self.in_proj(x.transpose(1, 2))

        # Full length crop
        B, T, _ = x.shape
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        # Conformer
        enc, _ = self.encoder(x, lengths)  

        # Heads
        onset_logits = self.onset_head(enc)   

        if teach_on is not None:
            #(B, 88, T) to (B, T, 88)
            on_cond = teach_on.transpose(1, 2)
        else:
            on_cond = torch.sigmoid(onset_logits)
        # (B, T, D)  
        h_cond = enc + self.frame_cond(on_cond)      
        frame_logits = self.frame_head(h_cond)

        # Offset/Velocity (B, T, 88)
        offset_logits = self.offset_head(enc)        
        vel_logits = self.vel_head(enc)         

        # Pedals (B, T, 1)
        ped64_logits  = self.ped64_head(enc)        
        ped67_logits  = self.ped67_head(enc)

        return onset_logits, frame_logits, offset_logits, vel_logits, ped64_logits, ped67_logits
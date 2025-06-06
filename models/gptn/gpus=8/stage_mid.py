import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import Block

class StageMid(nn.Module):

    def __init__(self, n_embd=384, n_head=6, block_size=256, dropout=0.2, n_layer=1):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input0):
        x = input0.clone()
        x = self.blocks(x) # (B,T,C)

        return x
    
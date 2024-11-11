import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dataset
import math

"""
Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(pe, (1,0,2))
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Base(nn.Module):
    def __init__(self, device,
                    max_len,
                    num_tokens,
                    dim=64,
                    nhead=8,
                    num_encoders=2,
                    num_decoders=2,
                    d_feedforward=1024,
                    dropout=0.1,
                    batch_first=True):
        super(Base, self).__init__()
        self.max_len = max_len
        self.device = device
        self.tokens = num_tokens
        ##Create encoder layers##
        self.src_emb = nn.Embedding(num_tokens, dim) 
        self.tgt_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = PositionalEncoding(dim, max_len=max_len)
        ##Create Transformer##
        self.transformer = nn.Transformer(d_model=dim, nhead=nhead,num_encoder_layers=num_encoders,     \
                                         num_decoder_layers=num_decoders, dim_feedforward=d_feedforward, \
                                         dropout=dropout, batch_first=batch_first)
        ##Create Final Linear Layer##
        self.policy_head = nn.Sequential(*[nn.Linear(dim, num_tokens), nn.ReLU(), nn.Linear(num_tokens, num_tokens)])
        self.value_head  = nn.Sequential(*[nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1)])

    def get_src_pad_mask(src, pad_idx):
        return src == pad_idx

    def forward(self, src, tgt, pad_idx, value=False):
        src_in = self.pos_emb(self.src_emb(src))
        tgt_in = self.pos_emb(self.tgt_emb(tgt))
        casual_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
        pad_mask = Base.get_src_pad_mask(src, pad_idx)
        trans_out = self.transformer(src=src_in, tgt=tgt_in, tgt_mask=casual_mask, \
                    src_key_padding_mask=pad_mask)
        if value:
            return self.policy_head(trans_out), self.value_head(trans_out)

        return self.policy_head(trans_out)

class clippedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model, model_old, eps, c_1, src, generated, rewards, pad_idx, eos_idx, device):

        model.eval()
        model_old.eval()

        N = generated.shape[0]
        S = generated.shape[1]

        #pad_mask = torch.logical_not(tgt == pad_idx).to(torch.int64).to(device)
        eos_mask  = (generated==eos_idx)
        first_eos = ((generated == eos_idx).cumsum(dim=1).cumsum(dim=1) == 1)
        eos_mask = torch.logical_xor(eos_mask.cumsum(dim=1), first_eos)
        eos_mask = torch.logical_not(eos_mask)

        policy, value = model(src, generated, pad_idx, value=True)
        policy_old    = model_old(src, generated, pad_idx)
        policy = F.softmax(policy, dim=1)
        policy_old = F.softmax(policy_old, dim=1)

        batch_index = torch.arange(N).unsqueeze(1).expand(N,S)
        state_index = torch.arange(S).expand(N,S)
        pi = policy[batch_index, state_index, generated]
        pi_old = policy_old[batch_index, state_index, generated]
        
        r = torch.exp(torch.log(pi) - torch.log(pi_old))
        r_clipped = torch.clamp(r, min=1-eps, max=1+eps)

        rewards = rewards.unsqueeze(1).expand(N,S)
        value = value.squeeze(-1)
        A = (rewards - value)

        #l_clip = (torch.min(r*A, r_clipped*A) * eos_mask)
        l_clip = torch.min(r*A, r_clipped*A) 
        l_clip = l_clip.sum()/(N*eos_mask.sum())

        rewards = rewards * eos_mask
        value = value * eos_mask
        rewards = rewards.reshape(-1,1)
        value   = value.reshape(-1,1)

        l_value = F.mse_loss(value, rewards)

        model.train()
        model_old.train()

        #print(l_value)
        #print(l_clip, l_value)
        #return l_value

        return -l_clip + c_1*l_value









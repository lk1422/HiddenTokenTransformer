import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import random

#Remove Later after generalizing
def generate_batched_sequence(model, src, device, dset, hidden_prob=0):
    #src (N, s)
    tgt = torch.ones(src.shape[0]).to(torch.int64) * dset.get_sos_idx()
    eos_detected = torch.zeros(src.shape[0]).to(torch.bool).to(device)
    tgt = tgt.to(device).unsqueeze(1)
    src = src.to(device)
    max_iterations = dset.get_max_len()
    it = 0
    while not torch.all(eos_detected) and it < max_iterations:
        policy = model(src, tgt, dset.get_pad_idx())
        next_token_policy = F.softmax(policy[:, -1], dim=-1)
        next_tokens = Categorical(next_token_policy).sample().unsqueeze(1)
        hidden_mask = (torch.rand(*next_tokens.shape) < hidden_prob).to(device)
        hidden_tokens = torch.ones_like(next_tokens) * dset.get_hidden_token()
        hidden_tokens = hidden_tokens.to(device)
        next_tokens   =  torch.logical_not(hidden_mask) * next_tokens + hidden_mask*hidden_tokens
        eos_detected  = eos_detected | (next_tokens == dset.get_eos_idx())
        tgt = torch.concat((tgt, next_tokens), dim=1)
        it+=1
    return tgt

def check_example(y_hat, tgt, dset):
    y_hat_ptr = 0
    tgt_ptr = 0
    hidden = False
    hidden_used = False

    while y_hat_ptr < y_hat.shape[0] and tgt_ptr < tgt.shape[0]:

        if y_hat[y_hat_ptr].item() == dset.get_eos_idx() and \
           tgt[tgt_ptr] == dset.get_eos_idx():
               return 2 if hidden_used else 1

        if y_hat[y_hat_ptr].item() == dset.get_eos_idx() and \
           tgt[tgt_ptr] == dset.get_eos_idx():
               return 1
        if hidden:
            if y_hat[y_hat_ptr] == dset.get_hidden_token():
                hidden = False
            y_hat_ptr +=1
            continue

        if y_hat[y_hat_ptr].item() == dset.get_hidden_token():
            hidden = True
            hidden_used = True
            y_hat_ptr+=1
            continue

        if y_hat[y_hat_ptr] != tgt[tgt_ptr]:
            return 0

        else:
            y_hat_ptr += 1
            tgt_ptr += 1

    return 0

def get_reward(y_hat, tgt, dset, device):
    reward = torch.zeros(tgt.shape[0])
    for b in range(tgt.shape[0]):
        reward[b] = check_example(y_hat[b], tgt[b], dset)
    return reward

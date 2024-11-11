import torch
import math
import os
import random
import torch.nn as nn
from models import Base
from train import train
import json

N_DIMS = ( 5, 8)
NHEADS = ( 3, 5)
D_FF   = ( 5,10)
LR     = (-4,-9)

#NOT TUNED PARAMETERS
EPOCHS     = 1024
N_ITS      = 512
BATCH_SIZE = 32
VERBOSE    = True


METRIC_FILE    = "/media/lenny/e8491f8e-2ac1-4d31-a37f-c115e009ec90/hidden_digits/logs/"
PARAMETER_FILE = "/media/lenny/e8491f8e-2ac1-4d31-a37f-c115e009ec90/hidden_digits/params/"

def train_custom(dset, device, n_layers, n_dims, n_heads, d_ff, lr, pth=None):
    name = f"model3_{n_layers}_{n_dims}_{n_heads}_{d_ff}_{lr}"
    metrics = {"loss":[], "eval":[]}
    model =  Base(device, 2*dset.get_max_len(), dset.get_num_tokens(), dim=n_dims, \
                  nhead=n_heads, num_encoders=n_layers, num_decoders=n_layers,    \
                  d_feedforward=d_ff).to(device)
    if pth!= None:
        model.load_state_dict(torch.load(PARAMETER_FILE+pth))

    learning_rate = lr
    loss = torch.nn.CrossEntropyLoss(ignore_index=dset.get_pad_idx())
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    train(model, device, EPOCHS, BATCH_SIZE, N_ITS, loss, optim, dset, metrics, \
        PARAMETER_FILE, name, verbose=VERBOSE, killable=False)

    with open(METRIC_FILE+name+".json", 'w') as f:
        json.dump(metrics, f)

def search_space(n_layers, n_models, device, dset):
    loss = torch.nn.CrossEntropyLoss(ignore_index=dset.get_pad_idx())
    for n in range(n_models):
        n_dims = int(math.pow(2, random.randint(*N_DIMS)))
        nheads = int(math.pow(2, random.randint(*NHEADS)))
        d_ff   = int(math.pow(2, random.randint(*D_FF)))
        for lr in range(LR[1], LR[0]+1):
            name = f"model_{n_layers}_{n_dims}_{nheads}_{d_ff}_{lr}"
            metrics = {"loss":[], "eval":[]}

            model =  Base(device, 2*dset.get_max_len(), dset.get_num_tokens(), dim=n_dims, \
                          nhead=nheads, num_encoders=n_layers, num_decoders=n_layers,    \
                          d_feedforward=d_ff).to(device)

            learning_rate = math.pow(10, lr)
            print(learning_rate)
            optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
            train(model, device, EPOCHS, BATCH_SIZE, N_ITS, loss, optim, dset, metrics, \
                PARAMETER_FILE, name, verbose=VERBOSE)

            with open(METRIC_FILE+name+".json", 'w') as f:
                json.dump(metrics, f)


def decode_file_name(filename):
    info = filename[6:-5].split("_")
    layers  = info[0]
    n_dims  = info[1]
    n_heads = info[2]
    d_ff    = info[3]
    lr      = info[4]
    return [("Layers", layers), ("N_dims", n_dims), ("N_heads", n_heads), ("D_ff", d_ff),\
            ("LR", lr)]
def top_n_params(n):
    files = os.listdir(METRIC_FILE)
    losses = []

    for file in files:
        if file[-4:] != "json":
            continue
        with open(os.path.join(METRIC_FILE, file), 'r') as f:
            metrics = json.load(f)
            loss = metrics["loss"][-1]
            losses.append([loss] + decode_file_name(file))
    return sorted(losses)[:n]






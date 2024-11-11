from dataset import Arithmetic
from models import Base, clippedLoss
from train import *
from hyperparam import search_space, top_n_params, train_custom
from util import *
import torch
import sys
import copy
import json

if torch.backends.mps.is_available():
    print("MPS")
    device = torch.device('mps')
elif torch.cuda.is_available():
    print("CUDA")
    device = torch.device('cuda')
else:
    print("WARNING CPU IN USE")
    device = torch.device('cpu')

dset = Arithmetic(max_val=1e4, test_data=True)
param_file = "/media/lenny/e8491f8e-2ac1-4d31-a37f-c115e009ec90/hidden_digits/params/saved/"
METRIC_FILE    = "/media/lenny/e8491f8e-2ac1-4d31-a37f-c115e009ec90/hidden_digits/logs/"

def test_generate_sequence():
    model = Base(device, 2*dset.get_max_len(), dset.get_num_tokens(), dim=256, \
            nhead=32, num_encoders=4, num_decoders=4, d_feedforward=1024)
    params = torch.load(param_file+"model3_4_256_32_1024_0.0001_epoch_1023.pth")
    model.load_state_dict(params)
    model = model.to(device)
    batch = dset.get_batch(2)
    out = generate_batched_sequence(model, batch[0], device, dset)
    print(dset.get_str(batch[0][0].tolist()))
    print(dset.get_str(out[0].tolist()))
    print(dset.get_str(batch[0][1].tolist()))
    print(dset.get_str(out[1].tolist()))
    print("REWARDS")
    print(get_reward(out, batch[1].to(device),dset,device))

def test_reward():
    y_hat = torch.tensor([1,2,3,dset.get_hidden_token(),3,4,dset.get_eos_idx(), 6])
    y= torch.tensor([1,2,3,4,dset.get_eos_idx()])
    print(y_hat)
    print(y)
    print(check_example(y_hat,y,dset))


def test_search_space():
    #FOR SAMIR TO RUN
    search_space(2, 10, device, dset)
    #FOR LAPTOP
    search_space(4, 10, device, dset)
    #FOR DESKTOP
    search_space(8, 10, device, dset)

def test_loss():

    model = Base(device, 2*dset.get_max_len(), dset.get_num_tokens(), dim=256, \
            nhead=32, num_encoders=4, num_decoders=4, d_feedforward=1024)
    params = torch.load(param_file+"model3_4_256_32_1024_0.0001_epoch_1023.pth")
    model.load_state_dict(params)
    model = model.to(device)
    model_old = copy.deepcopy(model)
    eps = 0.2
    src, tgt = dset.get_batch(2)
    src,tgt = src.to(device), tgt.to(device)
    gen = generate_batched_sequence(model, src, device, dset)
    reward = get_reward(gen, tgt, dset, device).to(device)
    loss = clippedLoss()
    pad_idx = dset.get_pad_idx()
    eos_idx = dset.get_eos_idx()
    print(loss(model, model_old, eps, src, gen, reward, pad_idx, eos_idx, device))

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def trainRL():
    model = Base(device, 2*dset.get_max_len(), dset.get_num_tokens(), dim=256, \
            nhead=32, num_encoders=4, num_decoders=4, d_feedforward=1024)
    #params = torch.load(param_file+"PPO_Value_Pretrained.pth")
    #model.load_state_dict(params)
    #unfreeze(model)
    model = model.to(device)
    epochs = 10000
    sub_epochs = 32
    batch_size = 128
    eps = 0.2
    optim = torch.optim.Adam(model.parameters(), lr=3e-5)
    name = "modelPPOFresh"
    metric = {"loss": [], "reward": []}
    c_1 = 0.5
    trainPPO(model, device, dset, epochs, sub_epochs, batch_size, eps, c_1, optim, metric, param_file, name, 150, 0.99)
    with open(METRIC_FILE+name+".json", 'w') as f:
        json.dump(metric, f)

trainRL()





#test_loss()
#test_search_space()
#print(top_n_params(4))
#test_generate_sequence()
#test_reward()

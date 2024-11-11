from .MDP import *
from PPO.base_model import *
import copy
from PPO.PPO import clippedLossSequential
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import *

device = torch.device('cuda')

def mainGridWorld():
    gw = GridWorld("testMDP/example.gw")
    model = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=16, num_encoders=1, \
            num_decoders=1, d_feedforward=32).to(device)
    x, tgt, rewards, acts = generate_sequence_grid_world(model, gw, 2, device)
    l_fn = clippedLossSequential(0.2, 1.0, 0.03, gw.PAD, device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses, rewards, succ = trainPPO(model, device, gw, 50, 128, 128, l_fn, optim)
    metrics = [losses, rewards, succ]
    pickle.dump(metrics, open("metrics.pkl", "wb"))

def step_through():
    gw = GridWorld("testMDP/easy.gw")
    np.random.seed(44)
    torch.random.manual_seed(44)

    model = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=16, num_encoders=1, \
            num_decoders=1, d_feedforward=32).to(device)
    model.load_state_dict(torch.load("testMDP/aux/param_2.pth"))
    model = model.to(device)
    model_old = copy.deepcopy(model).to(device)
    #model_old = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=128)
    model = model.to(device)
    src, gen, reward, acts = generate_sequence_grid_world(model, gw, 3, device)
    eos_mask = get_eos_mask(gw, gen).to(device)
    print(src)
    print(gen)
    print(reward)
    print(acts)
    print(eos_mask)
    l_fn = clippedLossSequential(0.3, 1, 0.01, gw.PAD, device)
    l, _ = l_fn(model, model_old, src, gen, acts, reward, eos_mask)
    print(l)


    d = torch.tensor(gw.destination).to(device)
    

def experiment():
    gw = GridWorld("testMDP/easy.gw")
    print(gw.destination_reward)
    model = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=16, num_encoders=1, \
            num_decoders=1, d_feedforward=32).to(device)
    model.load_state_dict(torch.load("testMDP/aux/param_4.pth"))
    model = model.to(device)
    src, gen, reward, acts = generate_sequence_grid_world(model, gw, 2, device)
    l_fn = clippedLossSequential(0.3, 1, 0.01, gw.PAD, device)
    eos_mask = get_eos_mask(gw, gen).to(device)[:, :-1]
    print(gen)
    print(l_fn.get_gae(reward,eos_mask))
    print(reward)
    src = torch.ones(1,1) * 11
    dec = torch.ones(1,1) * 11
    src = src.to(device).to(torch.int32)
    dec = dec.to(device).to(torch.int32)
    model.eval()
    print(src)
    print(dec)
    p, v = model(src, dec, -1, value=True)
    print(p)
    print(v)

def graph_metrics():
    loss, reward, precentage = pickle.load(open("testMDP/aux/metrics_4.pkl", "rb"))
    precentage = [prec.item() for prec in precentage]
    T = [i for i in range(len(loss))]
    fig, ax = plt.subplots(3)

    ax[0].plot(T, loss)
    ax[0].set_title("Loss Vs Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    ax[1].plot(T, reward)
    ax[1].set_title("Return Vs Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Return (Sum of Rewards)")

    ax[2].plot(T, precentage)
    ax[2].set_title("Success Rate Vs Epochs")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Success Rate (%)")

    plt.show()


if __name__ == "__main__":
    graph_metrics()

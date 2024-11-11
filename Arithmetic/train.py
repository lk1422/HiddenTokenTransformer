import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
import copy
from models import *

def get_accuracy(model, device, dset, batch_size, test=False):
    ##Fix to ignore Pad accuracy
    pad_idx = dset.get_pad_idx()
    x,y = dset.get_batch(batch_size, test=test)
    x,y = x.to(device), y.to(device)
    y_ = torch.concat((y[:, 1:], torch.ones(y.shape[0], 1).to(device) * pad_idx), dim=1)
    out = torch.argmax(model(x,y, pad_idx), dim=-1)
    return ((out == y_).sum() / (torch.ones_like(out).sum())).item()

def eval(model, device, dset, epochs, sub_epochs, batch_size):
    test_acc = get_accuracy(model, device, dset, 512, test=True), 
    train_acc = get_accuracy(model, device, dset, 512, test=True)
    return (test_acc, train_acc)

def trainPPO(model, device, dset, epochs, sub_epochs, batch_size, eps, c_1, optim, metric, path, name, hidden_step, hidden_decay):
    model_old = copy.deepcopy(model)
    loss = clippedLoss()
    hidden_prob = 0.01
    for e in range(epochs):
        src, tgt = dset.get_batch(batch_size)
        src,tgt = src.to(device), tgt.to(device)
        gen = generate_batched_sequence(model, src, device, dset, hidden_prob=hidden_prob)
        reward = get_reward(gen, tgt, dset, device).to(device)
        pad_idx = dset.get_pad_idx()
        eos_idx = dset.get_eos_idx()

        temp_model = copy.deepcopy(model)
        avg_loss = 0
        for s_e in range(sub_epochs):
            optim.zero_grad()
            l = loss(model, model_old, eps, c_1, src, gen, reward, pad_idx, eos_idx, device)
            l.backward()
            optim.step()
            avg_loss+=l.item()

        avg_loss = avg_loss/sub_epochs
        avg_reward = reward.sum()/batch_size
        metric["loss"].append(avg_loss)
        metric["reward"].append(avg_reward)
        print("EPOCH", e+1)
        print("Average Loss", avg_loss)
        print("Average Reward", avg_reward)
        print("="*10)

        if e % hidden_step == 0 and e != 0:
            hidden_prob*=hidden_decay
        if e % 100 == 0 and e != 0:
            torch.save(model.state_dict(), path+name+f"_epoch_{e}.pth")

        model_old = temp_model


def train(model, device, epoch, batch_size, n_iterations, loss, optim, dset, metrics, path, name, verbose=False, killable=True):
    pad_idx = dset.get_pad_idx()
    losses = []
    for e in range(epoch):
        avg_loss = 0
        for n in range(n_iterations):
            x, y = dset.get_batch(batch_size)
            x, y = x.to(device), y.to(device)
            y_ = torch.concat((y[:, 1:], torch.ones(y.shape[0], 1).to(device) * pad_idx), dim=1)
            optim.zero_grad()
            out = model(x, y, pad_idx)
            y_ = y_.reshape(-1).to(torch.int64)
            out = out.reshape(-1, out.shape[-1])
            l = loss(out, y_)
            avg_loss+=l.item()
            l.backward()
            optim.step()
        epoch_loss = avg_loss/n_iterations
        metrics["loss"].append(epoch_loss)
        model.eval()
        eval_metrics = eval(model, device, dset)
        metrics["eval"].append(eval_metrics)
        torch.save(model.state_dict(), path+name+f"_epoch_{e}.pth")
        model.train()
        if verbose:
            print(epoch_loss)
            print(eval_metrics)
            print("=="*10)
        if len(losses) != 3:
            losses.append(epoch_loss)
        if len(losses) == 3 and killable:
            if abs(losses[2] - losses[0]) < 0.1:
                print("KILLED")
                metrics["KILLED"] = True
                break

    if verbose:
        print("TRAINING COMPLETE")

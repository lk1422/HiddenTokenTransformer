import sys
import argparse
from torch.utils.tensorboard import SummaryWriter

from addition import AdditionDataset
from token_lookup import *
from models import *

def parse_args():
    parser = argparse.ArgumentParser(description="Parse Training Parameters")
    parser.add_argument("--seq2seq", type=bool, default=False)
    parser.add_argument("--d_model", type=float, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n_encoder", type=int, default=2)
    parser.add_argument("--n_decoder", type=int, default=2)
    parser.add_argument("--d_feedforward", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_iterations", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default = 100)
    parser.add_argument("--log_dir", type=str, default ="logs")
    parser.add_argument("--save_freq", type=int, default = 1000)

    return parser.parse_args()


def main():
    #Parse Which Model & Parameters to use
    args = parse_args()

    device = torch.device(args.device)
    max_seq_len = (3 * args.digits * 4) +  3

    if args.seq2seq:
        model = EncoderDecoderArithmetic( 
             args.d_model, 
             args.nhead, 
             args.n_encoder, 
             args.n_decoder, 
             args.d_feedforward,
             max_seq_len,
             device
        )
    else:
        model = EncoderArithmetic( 
             args.d_model, 
             args.nhead, 
             args.n_encoder, 
             args.d_feedforward,
             max_seq_len,
             device
        )
    #print(args.seq2seq)
    #print(type(model))
    #Write Train Loop
    dataset = AdditionDataset(max_digits=args.digits)
    train(model, dataset, args, device, args.seq2seq)
    #TODO Log training information with tensorboard


def acc_helper(prediction, tgt):
    digit_count = 0
    correct_digits = 0
    correct_wholes = 0
    for i in range(tgt.shape[0]):
        correct = True
        for j in range(tgt.shape[1]):

            if tgt[i][j] == TOKEN_LOOKUP["<EOS>"]:
                break
            digit_count += 1

            if tgt[i][j] == prediction[i][j]:
                correct_digits += 1
            else:
                correct = False

        if correct:
            correct_wholes += 1
    return (correct_digits / digit_count), (correct_wholes / tgt.shape[0])



def get_accuracy(model, dataset, args, device, seq2seq):
        #GET DIGIT BY DIGIT ACCURACY AND TOTAL ACCURACY
        seq_len = None if seq2seq else 3*args.digits*4 + 3
        x, y = dataset.generate_batch(100, seq_len=seq_len, seq2seq=seq2seq)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if args.seq2seq:
                tgt_in = y[:, :-1]
                tgt_out = y[:, 1:]
                out = model(x, tgt_in)
            else:
                out = model(x)
                tgt_out = y

        out = out.transpose(0, 1)
        #print(out)
        prediction = torch.argmax(out, dim=-1)
        #print("TGT", tgt_out)
        #print(out)
        #print(prediction)
        return acc_helper(prediction, tgt_out)

def train(model, dataset, args, device, seq2seq):
    model = model.to(device)
    loss = nn.CrossEntropyLoss(ignore_index=TOKEN_LOOKUP["<PAD>"])
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    seq_len = None if seq2seq else 3*args.digits*4 + 3

    writer = SummaryWriter(args.log_dir)

    accumulated_loss = 0
    for i in range(args.n_iterations):
        x, y = dataset.generate_batch(args.batch_size, seq_len=seq_len, seq2seq=seq2seq)
        x, y = x.to(device), y.to(device)
        #print(x)
        #print(y)
        optim.zero_grad()
        if args.seq2seq:
            #y: [T, N, E]
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            out = model(x, tgt_in)
        else:
            out = model(x)
            tgt_out = y

        out = out.transpose(0, 1)
        #print(out.shape)
        #print(tgt_out.shape)
        tgt_out = tgt_out.reshape(-1)
        out = out.reshape(-1, len(TOKEN_LOOKUP))

        l = loss(out, tgt_out)
        accumulated_loss += l.item()
        l.backward()
        optim.step()

        if (i+1) % args.log_freq == 0:
            digit_acc, total_acc = get_accuracy(model, dataset, args, device, seq2seq)
            writer.add_scalar("Loss/Iteration", accumulated_loss/ args.log_freq, i)
            writer.add_scalar("Digit Accuracy/Iteration", digit_acc, i)
            writer.add_scalar("Total Accuracy/Iteration", total_acc, i)
            print("AVG LOSS", accumulated_loss/ args.log_freq)
            print("DIG ACC", digit_acc)
            print("TOT ACC", total_acc)
            for name, param in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", param, i)
                writer.add_histogram(f"Gradients/{name}", param.grad, i)

            accumulated_loss = 0

        if (i+1) % args.save_freq == 0:
            torch.save(model.state_dict(), f"params/model_parameters_{i}.pth")


    writer.close()

if __name__ == "__main__":
    main()

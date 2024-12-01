from train import acc_helper
from addition import AdditionDataset
from token_lookup import *
from models import *

max_seq_len = (3 * 1 * 4) +  3
device = torch.device('cpu')

model = EncoderDecoderArithmetic( 128, 4, 
     2, 
     2,
     1024,
     max_seq_len,
     device
 )
model.load_state_dict(torch.load("params/model_parameters_2999.pth"))
dataset = AdditionDataset(4)

def get_accuracy(model, dataset, seq_len, device):
        #GET DIGIT BY DIGIT ACCURACY AND TOTAL ACCURACY
        x, y = dataset.generate_batch(5, seq_len=seq_len, seq2seq=True)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            out = model(x, tgt_in)

        out = out.transpose(0, 1)
        prediction = torch.argmax(out, dim=-1)
        return acc_helper(prediction, tgt_out)

def stringify(x):

    return "".join([REVERSE_LOOKUP[item.item()] for item in x[0]])
    

def render_example(model, dataset, seq_len, device):
        x, y = dataset.generate_batch(1, seq_len=seq_len, seq2seq=True)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            out = model(x, tgt_in)

        out = out.transpose(0, 1)
        prediction = torch.argmax(out, dim=-1)
        return f"{stringify(x)}\n{stringify(prediction)}\nExpected: {stringify(y)}"

for _ in range(10):
    print(render_example(model, dataset, max_seq_len, device))
    print("="*10)

from train import acc_helper
from addition import AdditionDataset
from token_lookup import *
from models import *

max_seq_len = (3 * 1 * 4) +  3
device = torch.device('cpu')
seq2seq=False

model = EncoderDecoderArithmetic( 128, 4, 
     2, 
     2,
     256,
     max_seq_len,
     device
 )

model.load_state_dict(torch.load("params/model_parameters_999.pth"))
print("NUM PARAMS", sum(p.numel() for p in model.parameters()))
dataset = AdditionDataset(4)

def solve(problem, model):
    tokens = torch.tensor([TOKEN_LOOKUP[item] for item in problem]).unsqueeze(0)
    tgt = [TOKEN_LOOKUP["<SOS>"]]
    index = 0
    while True:
        tgt_tensor = torch.tensor(tgt).unsqueeze(0)
        out = model(tokens, tgt_tensor)
        next_token = torch.argmax(out[index][0]).item()
        if next_token == TOKEN_LOOKUP["<EOS>"]:
            break
        tgt.append(next_token)
        index += 1
    return "".join([REVERSE_LOOKUP[tok] for tok in tgt])

def get_accuracy(model, dataset, seq_len, device):
        #GET DIGIT BY DIGIT ACCURACY AND TOTAL ACCURACY
        x, y = dataset.generate_batch(100, seq_len=seq_len, seq2seq=True)
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
        x, y = dataset.generate_batch(1, seq_len=None, seq2seq=True)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            tgt_in = y[:, :-1]
            tgt_out = y[:, 1:]
            out = model(x, tgt_in)

        out = out.transpose(0, 1)
        prediction = torch.argmax(out, dim=-1)
        return f"{stringify(x)}\n{stringify(prediction)}\nExpected: {stringify(y)}"




print("ACC", get_accuracy(model, dataset, None, device))
print("4013+620=", solve("4013+620=", model))
#print("ACC", get_accuracy(model, dataset, max_seq_len, device))
for _ in range(10):
    print(render_example(model, dataset, max_seq_len, device))
    print("="*10)





import torch
import random
from typing import List
from torch.nn.utils.rnn import pad_sequence


SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
HIDDEN_START = "<S_HIDDEN>"
HIDDEN_END   = "<E_HIDDEN>"

class Arithmetic:
    def __init__(self, max_val=1e10, test_data=True):
        #ADD DIGITS
        self.TOKENS = {str(i): i for i in range(0,10)}

        #ADD OPERATORS
        self.TOKENS["+"] = 10
        self.TOKENS["-"] = 11
        self.TOKENS["*"] = 12

        #ADD UTILITY TOKENS
        self.TOKENS["<EOS>"] = 13
        self.TOKENS["<PAD>"] = 14
        self.TOKENS["<SOS>"] = 15
        self.TOKENS[HIDDEN_START] = 16
        self.TOKENS[HIDDEN_END] = 17

        #CREATE REVERSE MAPPING
        self.VOCAB = {value:key_ for (key_,value) in self.TOKENS.items()}

        #STORE THE LARGETS POSSIBLE ARITHMETIC RESULTING VALUE
        self.maximum_operand = max_val
        self.maximum_result = max_val**2+1
        self.max_len = len(str(int(self.maximum_operand)))*2 + 6
        self.num_tokens = len(self.TOKENS)
        self.test_data = test_data
        ##Add Tests for generalization to larger numbers
        if(test_data):
            self.maximum_operand = int(self.maximum_operand/2)

    def get_max_len(self):
        return self.max_len

    def get_pad_idx(self):
        return self.TOKENS["<PAD>"]

    def get_sos_idx(self):
        return self.TOKENS["<SOS>"]

    def get_eos_idx(self):
        return self.TOKENS["<EOS>"]
    
    def get_hidden_token(self):
        return self.TOKENS[HIDDEN_START]

    def get_num_tokens(self):
        return self.num_tokens


    def get_batch(self, batch_size, test=False):
        """
        Generate a random arithmetic statements for the batch
        """
        expressions = []
        results = []

        for _ in range(batch_size):
            exp, res = self.generate_expression(test)
            expressions.append(torch.tensor(exp))
            results.append(torch.tensor(res))

        padded_exp = pad_sequence(expressions, padding_value=self.TOKENS["<PAD>"], batch_first=True).to(torch.int64)
        padded_res = pad_sequence(results, padding_value=self.TOKENS["<PAD>"], batch_first=True).to(torch.int64)

        return  padded_exp, padded_res

    def generate_expression(self, test=False):
        if test and self.test_data:
            """Add case to generate specific test data"""
            operand1 = random.randint(int(-self.maximum_operand) , int(self.maximum_operand) )
            operand2 = random.randint(int(-self.maximum_operand) , int(self.maximum_operand) )
        else:
            operand1 = random.randint(int(-self.maximum_operand/2) , int(self.maximum_operand/2) )
            operand2 = random.randint(int(-self.maximum_operand/2) , int(self.maximum_operand/2) )
        operation = "*"
        #operation = random.choice(["+", "-", "*"])
        #operation = random.choice(["+", "-"])

        result = 0
        if operation == "+":
            result = operand1 + operand2
        elif operation == "-":
            result = operand1 - operand2
        elif operation == "*":
            result = operand1 * operand2

        result = str(int(result))
        expression = str(int(operand1)) + operation + str(int(operand2))
        return  self.tokenize_expression(expression), self.tokenize_expression(result)

    def tokenize_expression(self, exp:str):
        tokens = [self.TOKENS[c] for c in exp]
        tokens.append(self.TOKENS[EOS])
        tokens = [self.TOKENS[SOS]] + tokens

        return tokens

    def get_tensor(self, exp):
        tok = self.tokenize_expression(exp)
        return torch.tensor(tok).unsqueeze(0)


    def get_str(self, tokens:List[int] ):
        digits = [self.VOCAB[t] for t in tokens]
        return "".join(digits)



import numpy as np
from token_lookup import TOKEN_LOOKUP, REVERSE_LOOKUP
import torch

# device = th.device("cuda" if th.cuda.is_available() else "cpu")

class AdditionDataset():
    def __init__(self, max_digits=2):
        self.max_digits = max_digits
        self.test_digit_length = 3*self.max_digits
        self.max_len = self.test_digit_length * 3 + 3


    def _generate_problem(self):
        """Generate a single random problem."""
        num1 = np.random.randint(0, 10 ** self.max_digits)
        num2 = np.random.randint(0, 10 ** self.max_digits)
        problem = f"{num1}+{num2}="
        solution = str(num1 + num2)
        return problem, solution

    def _supervised_problem(self, seq2seq=False):
        problem, solution = self._generate_problem()
        if not seq2seq:
            return problem, solution
        return ["<SOS>"] + solution + ["<EOS>"]

    def generate_batch(self, batch_size, seq_len=None):
        problems = []
        solutions = []
        for _ in range(batch_size):
            p, s = self._supervised_problem()
            problems.append(self._encode_problem(p))
            solutions.append(self._encode_problem(s))
        problems = self._pad_sequence(problems, seq_len)
        solutions = self._pad_sequence(solutions, seq_len)
        return torch.tensor(problems, dtype=torch.int32), torch.tensor(solutions, dtype=torch.int64)

    @staticmethod
    def _pad_sequence(sequence, seq_len):
        new_sequence = []
        if seq_len == None:
            max_len = max([len(sub_seq) for sub_seq in sequence])
            for seq in sequence:
                seq_prime = seq + [TOKEN_LOOKUP["<PAD>"]] * (max_len - len(seq))
                new_sequence.append(seq_prime)
        else:
            for seq in sequence:
                seq_prime = seq + [TOKEN_LOOKUP["<PAD>"]] * (seq_len - len(seq))
                new_sequence.append(seq_prime)
        return new_sequence

    def _encode_problem(self, problem):
        """Encode a problem string into a numerical sequence."""
        return [TOKEN_LOOKUP[char] for char in problem]


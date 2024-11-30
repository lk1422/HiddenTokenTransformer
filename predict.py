import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch import optim
from torch.distributions import Categorical
from torch.nn import Embedding
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate
class OurEmbedding:
    def __init__(self, device):
        self.token_to_index = {
            "<sos>": 0,
            "<eos>": 1,
            "roses": 2,
            "are": 3,
            "red": 4
        }
        self.vocab_size = len(self.token_to_index)
        self.token_vec_dim = 8
        self.device = device
        self.torch_embedding = Embedding(self.vocab_size, self.token_vec_dim, device=self.device)


class PolicyNetwork(nn.Module):
    def __init__(self, our_embedding, device):
        super().__init__()
        self.our_embedding = our_embedding
        self.device = device
        self.transformer = nn.Transformer(
            d_model=self.our_embedding.token_vec_dim,
            nhead=8 if self.our_embedding.token_vec_dim % 8 == 0 else 2,
            batch_first=True,
            device=self.device
        )
        self.fc = nn.Sequential(
            nn.Linear(self.our_embedding.token_vec_dim, self.our_embedding.token_vec_dim * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.our_embedding.token_vec_dim * 2, self.our_embedding.vocab_size, device=self.device)
        )

    def forward(self, src_token_vecs, target_token_vecs):
        transformer_output = self.transformer(src_token_vecs, target_token_vecs)
        last_token_vec = transformer_output[:, -1, :]
        return self.fc(last_token_vec)

def predict_sentences(model_path="ppo_model.pt", num_sentences=10, max_length=10, temperature=1.0):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    our_embedding = OurEmbedding(device)

    # Initialize networks
    policy_network = PolicyNetwork(our_embedding, device)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    policy_network.load_state_dict(checkpoint['policy_network'])
    print(f"Model loaded from {model_path}")

    # Generate sentences
    index_to_token = {v: k for k, v in our_embedding.token_to_index.items()}
    generated_sentences = []

    for _ in range(num_sentences):
        src_token_indexes = [2, 3]  # Starting source tokens
        src_token_vecs = our_embedding.torch_embedding(torch.LongTensor([src_token_indexes]).to(device))

        target_token_indexes = [0]  # Start with <sos>
        for _ in range(max_length):
            target_token_vecs = our_embedding.torch_embedding(torch.LongTensor([target_token_indexes]).to(device))
            logits = policy_network(src_token_vecs, target_token_vecs)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Probabilistic sampling
            next_token = torch.multinomial(probs, num_samples=1).item()

            target_token_indexes.append(next_token)
            if next_token == 1:  # Stop if <eos> token is generated
                break

        # Convert token indexes to words
        sentence = " ".join([index_to_token[idx] for idx in target_token_indexes])
        generated_sentences.append(sentence)

    # Print generated sentences
    for i, sentence in enumerate(generated_sentences, 1):
        print(f"Sentence {i}: {sentence}")


# Call the prediction function with probabilistic sampling
predict_sentences(temperature=2)

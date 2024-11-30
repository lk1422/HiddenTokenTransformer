import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import PPO
from torchrl.data import TensorDict, ReplayBuffer

# Define the autoregressive transformer model
class AutoRegressiveTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_seq):
        embedded_input = self.embedding(input_seq)
        transformer_out = self.transformer(embedded_input, embedded_input)
        logits = self.fc(transformer_out)
        return logits


# Custom text prediction environment for "Roses are Red"
class RosesAreRedEnv(EnvBase):
    def __init__(self, tokenizer, max_seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.target_tokens = tokenizer.encode("Roses are Red")
        self.max_seq_len = max_seq_len
        self.reset()

    def step(self, action):
        # Compare predicted token to the target token
        predicted_token = action.item()
        correct_token = self.target_tokens[len(self.current_tokens)]
        reward = 1.0 if predicted_token == correct_token else -1.0
        self.current_tokens.append(predicted_token)

        # Check if sequence is complete
        done = len(self.current_tokens) >= len(self.target_tokens)
        obs = torch.tensor(self.current_tokens[-self.max_seq_len :]).unsqueeze(0)
        return TensorDict(
            {"obs": obs, "reward": reward, "done": done}, batch_size=[1]
        )

    def reset(self):
        # Start with a partial sequence
        self.current_tokens = self.target_tokens[:2]  # Provide a few starting tokens
        obs = torch.tensor(self.current_tokens).unsqueeze(0)
        return TensorDict({"obs": obs}, batch_size=[1])


# Hyperparameters
vocab_size = 1000
embed_dim = 128
num_heads = 4
num_layers = 2
max_seq_len = 10
lr = 0.001
num_epochs = 10
batch_size = 16

# Initialize components
tokenizer = ...  # Assume a tokenizer is provided
env = RosesAreRedEnv(tokenizer, max_seq_len)
model = AutoRegressiveTransformer(vocab_size, embed_dim, num_heads, num_layers)

# Define actor and critic
actor = ProbabilisticActor(
    module=model,
    distribution_class=Categorical,
)
critic = ValueOperator(module=nn.Linear(embed_dim, 1))

# Optimizer and replay buffer
optimizer = optim.Adam(model.parameters(), lr=lr)
replay_buffer = ReplayBuffer(storage=torch.empty(10000, dtype=torch.float))

# PPO training
ppo = PPO(actor, critic, optimizer, gamma=0.99, batch_size=batch_size)

for epoch in range(num_epochs):
    replay_buffer.clear()
    for step in range(100):  # Collect training data
        obs = env.reset()
        for _ in range(len(env.target_tokens) - 2):  # Limit to target length
            td = actor(obs)
            td["reward"], td["done"], td["next_obs"] = env.step(td["action"])
            replay_buffer.add(td)
            if td["done"].item():
                break
            obs = td["next_obs"]

    # Train with PPO
    for batch in replay_buffer.sample(batch_size):
        loss = ppo(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} complete.")

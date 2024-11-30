import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import random

class AdditionTokenizer:
    def __init__(self, max_num=99):
        self.max_num = max_num
        self.special_tokens = {
            'PAD': 0,
            'START': 1,
            'EOS': 2
        }
        
        # Create token mappings
        self.token_to_idx = {
            **{str(i): i+3 for i in range(max_num+1)},
            **self.special_tokens,
            '+': max_num + 4
        }
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        
        self.vocab_size = len(self.token_to_idx)
        self.pad_idx = self.special_tokens['PAD']
        self.start_idx = self.special_tokens['START']
        self.eos_idx = self.special_tokens['EOS']

    def encode_addition_problem(self, problem):
        tokens = [self.token_to_idx['START']]
        for char in problem:
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
        tokens.append(self.eos_idx)
        return tokens

    def encode_solution(self, solution):
        tokens = [self.token_to_idx['START']]
        for char in str(solution):
            tokens.append(self.token_to_idx[char])
        tokens.append(self.eos_idx)
        return tokens

    def decode_tokens(self, tokens):
        # Handle both tensor and list inputs
        if torch.is_tensor(tokens):
            tokens = tokens.cpu().numpy().tolist()
        
        # Filter out special tokens and convert to string
        filtered_tokens = [t for t in tokens if t not in [self.pad_idx, self.start_idx, self.eos_idx]]
        
        try:
            return ''.join([self.idx_to_token[t] for t in filtered_tokens])
        except KeyError as e:
            print(f"Decoding error. Problematic token: {e}")
            return ''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.permute(pe, (1,0,2))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerAdditionModel(nn.Module):
    def __init__(self, tokenizer, device, dim=64, nhead=8, num_encoders=2, num_decoders=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        
        # Embeddings
        self.src_emb = nn.Embedding(tokenizer.vocab_size, dim)
        self.tgt_emb = nn.Embedding(tokenizer.vocab_size, dim)
        self.pos_emb = PositionalEncoding(dim)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=dim, 
            nhead=nhead,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(dim, tokenizer.vocab_size),
            nn.ReLU(),
            nn.Linear(tokenizer.vocab_size, tokenizer.vocab_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
    
    def forward(self, src, tgt):
        # Embed and add positional encoding
        src_emb = self.pos_emb(self.src_emb(src))
        tgt_emb = self.pos_emb(self.tgt_emb(tgt))
        
        # Generate causal mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        
        # Transformer forward pass
        trans_out = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask)
        
        # Policy and value heads
        policy_logits = self.policy_head(trans_out)
        value_est = self.value_head(trans_out)
        
        return policy_logits, value_est

class PPOTrainer:
    def __init__(self, model, tokenizer, lr=1e-4, clip_range=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def generate_problem_batch(self, batch_size=32):
        problems = []
        solutions = []
        true_solutions = []
        for _ in range(batch_size):
            # Generate random addition problems
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            problem = f"{a}+{b}="
            solution = a + b
            
            # Encode
            prob_tokens = self.tokenizer.encode_addition_problem(problem)
            sol_tokens = self.tokenizer.encode_solution(solution)
            
            problems.append(prob_tokens)
            solutions.append(sol_tokens)
            true_solutions.append(solution)
        
        # Pad sequences
        max_prob_len = max(len(p) for p in problems)
        max_sol_len = max(len(s) for s in solutions)
        
        padded_problems = [p + [self.tokenizer.pad_idx] * (max_prob_len - len(p)) for p in problems]
        padded_solutions = [s + [self.tokenizer.pad_idx] * (max_sol_len - len(s)) for s in solutions]
        
        return (
            torch.tensor(padded_problems),
            torch.tensor(padded_solutions),
            true_solutions
        )
    
    def compute_returns_and_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Compute generalized advantage estimation (GAE)
        
        Args:
        - rewards: Tensor of rewards
        - values: Tensor of estimated values
        - gamma: Discount factor
        - lam: GAE smoothing parameter
        
        Returns:
        - returns: Discounted cumulative rewards
        - advantages: Generalized advantage estimation
        """
        # Ensure rewards and values are 1D tensors
        rewards = rewards.flatten()
        values = values.flatten()
        
        returns = torch.zeros_like(rewards, dtype=torch.float)
        advantages = torch.zeros_like(rewards, dtype=torch.float)
        
        # Compute returns (cumulative discounted rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + gamma * returns[t+1]
        
        # Compute advantages using GAE
        delta = rewards + gamma * torch.cat([values[1:], torch.zeros_like(values[-1:])]) - values
        advantages = self.discount_cumsum(delta, gamma * lam)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def discount_cumsum(self, x, discount):
        """
        Compute discounted cumulative sum of x
        
        Args:
        - x: Input sequence
        - discount: Discount factor
        
        Returns:
        - Discounted cumulative sum
        """
        cumsum = torch.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            cumsum[t] = x[t] + discount * cumsum[t+1]
        return cumsum
    
    def compute_rewards(self, problems, solutions, true_solutions):
        """
        Compute rewards based on prediction accuracy
        
        Args:
        - problems: Input problem tokens
        - solutions: Predicted solution tokens
        - true_solutions: Actual solutions
        
        Returns:
        - Rewards tensor
        """
        rewards = []
        for pred_sol, true_sol in zip(solutions, true_solutions):
            # Decode predicted solution
            pred_str = self.tokenizer.decode_tokens(pred_sol)
            try:
                pred_num = int(pred_str)
                # Reward based on accuracy
                if pred_num == true_sol:
                    reward = 1.0  # Full reward for exact match
                else:
                    # Partial reward based on proximity
                    reward = max(0, 1 - abs(pred_num - true_sol) / max(true_sol, 1))
            except (ValueError, TypeError):
                reward = 0.0  # Invalid prediction
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float)
    
    def ppo_update(self, problems, solutions, true_solutions, epochs=4):
        """
        Perform PPO update
        
        Args:
        - problems: Input problem tokens
        - solutions: Predicted solution tokens
        - true_solutions: Actual solutions
        - epochs: Number of optimization epochs
        """
        # Compute initial log probabilities and values
        with torch.no_grad():
            old_policy_logits, old_values = self.model(problems, solutions)
            old_log_probs = F.log_softmax(old_policy_logits, dim=-1)
            old_log_probs = old_log_probs.gather(-1, solutions.unsqueeze(-1)).squeeze(-1)
        
        # Compute rewards
        rewards = self.compute_rewards(problems, solutions, true_solutions)
        
        # Compute returns and advantages
        # Ensure old_values is squeezed correctly
        returns, advantages = self.compute_returns_and_advantages(rewards, old_values.squeeze())
        
        # PPO optimization loop
        for _ in range(epochs):
            # Recompute policy and values
            policy_logits, values = self.model(problems, solutions)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            log_probs = log_probs.gather(-1, solutions.unsqueeze(-1)).squeeze(-1)
            
            # Compute ratios and surrogate loss
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Ensure values and returns have compatible shapes
            values_squeezed = values.squeeze(-1)
            if values_squeezed.ndim > returns.ndim:
                values_squeezed = values_squeezed.reshape(returns.shape)
            
            # Value loss
            value_loss = F.mse_loss(values_squeezed, returns)
            
            # Entropy bonus
            entropy_loss = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(-1).mean()
            
            # Total loss
            loss = (policy_loss + 
                    self.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy_loss)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        return loss.item()

def train_addition_transformer():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AdditionTokenizer()
    model = TransformerAdditionModel(tokenizer, device).to(device)
    trainer = PPOTrainer(model, tokenizer)
    
    # Training loop
    losses = []
    for epoch in range(1000):  # Adjust number of epochs as needed
        # Generate batch of problems
        problems, solutions, true_solutions = trainer.generate_problem_batch()
        problems = problems.to(device)
        solutions = solutions.to(device)
        
        # PPO update
        loss = trainer.ppo_update(problems, solutions, true_solutions)
        losses.append(loss)
        
        # Optional: Evaluation/logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Average Loss = {np.mean(losses[-100:])}")
            
            # Sample problem evaluation
            test_problem = "12+34="
            test_prob_tokens = torch.tensor(tokenizer.encode_addition_problem(test_problem)).unsqueeze(0).to(device)
            test_sol_tokens = torch.tensor([tokenizer.start_idx]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_logits, _ = model(test_prob_tokens, test_sol_tokens)
                pred_token = torch.argmax(pred_logits[0, -1, :]).cpu().item()
                pred_solution = tokenizer.decode_tokens([pred_token])
                print(f"Test Problem: {test_problem}, Predicted: {pred_solution}")

# Main execution
if __name__ == "__main__":
    train_addition_transformer()

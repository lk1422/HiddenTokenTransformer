import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
MAX_ADDITION_VALUE = 999
MAX_OUTPUT_LENGTH = 5
VOCAB_SIZE = 12  # 0-9 digits, '+', '='

class PPOAutoregressiveAdditionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=1024, max_len=20):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Policy head (output projection)
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )
        
    def forward(self, input_seq, target_seq=None):
        batch_size, input_seq_len = input_seq.shape
        
        # Create input embeddings with positional encoding
        x = self.embedding(input_seq)
        x = x + self.positional_encoding[:, :input_seq_len, :]
        x = x.permute(1, 0, 2)  # (input_seq_len, batch_size, d_model)
        
        # Create memory for decoder (encoder representation)
        memory = x
        
        # Inference mode (autoregressive generation)
        if target_seq is None:
            generated_seq = []
            current_token = torch.full((batch_size, 1), fill_value=11, 
                                        dtype=torch.long, device=input_seq.device)  # Start with '=' token
            values = []
            
            for _ in range(MAX_OUTPUT_LENGTH):
                # Embed and encode current sequence
                current_emb = self.embedding(current_token)
                current_emb = current_emb + self.positional_encoding[:, :current_emb.shape[1], :]
                current_emb = current_emb.permute(1, 0, 2)
                
                # Generate next token
                decoder_output = self.transformer_decoder(current_emb, memory)
                decoder_output = decoder_output.permute(1, 0, 2)
                
                # Get logits and value
                logits = self.policy_head(decoder_output[:, -1:, :])
                value = self.value_head(decoder_output[:, -1:, :])
                values.append(value)
                
                # Sample or take argmax
                next_token = torch.multinomial(F.softmax(logits.squeeze(), dim=-1), num_samples=1)
                generated_seq.append(next_token)
                
                current_token = torch.cat([current_token, next_token], dim=1)
            
            return (torch.cat(generated_seq, dim=1), 
                    torch.cat(values, dim=1))
        
        # Training mode with teacher forcing
        else:
            # Prepare target sequence
            target_emb = self.embedding(target_seq)
            target_emb = target_emb + self.positional_encoding[:, :target_seq.shape[1], :]
            target_emb = target_emb.permute(1, 0, 2)
            
            # Create target mask to prevent attending to future tokens
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq.shape[1]).to(input_seq.device)
            
            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(target_emb, memory, tgt_mask=tgt_mask)
            decoder_output = decoder_output.permute(1, 0, 2)
            
            # Compute policy logits and values
            logits = self.policy_head(decoder_output)
            values = self.value_head(decoder_output)
            
            return logits, values

class PPOTrainer:
    def __init__(self, model, vocab_size, lr=3e-4, gamma=0.99, lambda_=0.95, 
                 clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.vocab_size = vocab_size

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Compute returns and GAE (Generalized Advantage Estimation)
        """
        returns = []
        advantages = []
        gae = 0
        
        # Reverse iteration for GAE
        for t in reversed(range(len(rewards))):
            # Compute delta
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state value
                delta = rewards[t] - values[t]
            else:
                next_value = values[t + 1]
                delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE computation
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
            
            # Compute returns (cumulative discounted rewards)
            returns.insert(0, gae + values[t])
        
        return torch.tensor(returns), torch.tensor(advantages)

    def train(self, input_ids, input_actions, rewards, old_log_probs, values):
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards, values, [1] * len(rewards))
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute new logits and values
        new_logits, new_values = self.model(input_ids, input_actions)
        
        # Reshape for computation
        new_logits = new_logits.reshape(-1, self.vocab_size)
        new_values = new_values.reshape(-1)
        input_actions = input_actions.reshape(-1)
        
        # Compute policy distribution
        dist = torch.distributions.Categorical(logits=new_logits)
        
        # Compute log probabilities of actions
        new_log_probs = dist.log_prob(input_actions)
        
        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy loss for exploration
        entropy_loss = dist.entropy().mean()
        
        # Composite loss
        loss = (policy_loss + 
                self.value_loss_coef * value_loss - 
                self.entropy_coef * entropy_loss)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }

class AdditionDataset:
    def __init__(self, max_value=999, num_samples=10000):
        self.samples = []
        for _ in range(num_samples):
            num1 = np.random.randint(0, max_value)
            num2 = np.random.randint(0, max_value)
            input_str = f"{num1}+{num2}="
            target_str = str(num1 + num2)
            self.samples.append((input_str, target_str))
    
    def tokenize(self, input_str, target_str):
        # Tokenize input and target strings
        input_tokens = [int(char) if char.isdigit() else 10 if char == '+' else 11 for char in input_str]
        target_tokens = [int(char) for char in target_str]
        
        # Pad sequences to max length
        input_tokens = input_tokens[:MAX_OUTPUT_LENGTH]
        target_tokens = target_tokens[:MAX_OUTPUT_LENGTH]
        input_tokens += [0] * (MAX_OUTPUT_LENGTH - len(input_tokens))
        target_tokens += [0] * (MAX_OUTPUT_LENGTH - len(target_tokens))
        
        return (torch.tensor(input_tokens), torch.tensor(target_tokens))

def compute_reward(predicted_str, target_str):
    """
    Compute reward based on digit-wise accuracy
    """
    # Remove leading zeros for fair comparison
    predicted_str = predicted_str.lstrip('0')
    target_str = target_str.lstrip('0')
    
    # Exact match
    if predicted_str == target_str:
        return 1.0
    
    # Partial accuracy
    correct_digits = sum(p == t for p, t in zip(predicted_str, target_str))
    return correct_digits / max(len(predicted_str), len(target_str))

def train_ppo_model(model, trainer, dataset, num_epochs=50, batch_size=64):
    train_metrics = {
        'policy_losses': [],
        'value_losses': [],
        'entropy_losses': [],
        'rewards': []
    }
    
    for epoch in range(num_epochs):
        epoch_metrics = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'rewards': []
        }
        
        # Sample batch
        batch_indices = np.random.choice(len(dataset.samples), batch_size)
        
        for idx in batch_indices:
            input_str, target_str = dataset.samples[idx]
            input_tokens, target_tokens = dataset.tokenize(input_str, target_str)
            
            # Generate sequence and get values
            with torch.no_grad():
                generated_tokens, values = model(input_tokens.unsqueeze(0))
            
            # Convert to strings for reward computation
            predicted_str = ''.join(map(str, generated_tokens.squeeze().tolist()))
            
            # Compute reward
            reward = compute_reward(predicted_str, target_str)
            
            # Compute log probabilities of original generation
            dist = torch.distributions.Categorical(logits=model(input_tokens.unsqueeze(0))[0].squeeze())
            old_log_probs = dist.log_prob(generated_tokens.squeeze())
            
            # PPO training step
            metrics = trainer.train(
                input_tokens.unsqueeze(0), 
                generated_tokens, 
                [reward], 
                old_log_probs,
                values.squeeze()
            )
            
            # Store metrics
            for key in metrics:
                epoch_metrics[key].append(metrics[key])
            epoch_metrics['rewards'].append(reward)
        
        # Compute and store average metrics
        for key in epoch_metrics:
            train_metrics[key].append(np.mean(epoch_metrics[key]))
        
        # Periodic logging
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: "
                  f"Avg Reward: {train_metrics['rewards'][-1]:.4f}, "
                  f"Policy Loss: {train_metrics['policy_losses'][-1]:.4f}, "
                  f"Value Loss: {train_metrics['value_losses'][-1]:.4f}")
    
    return train_metrics

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    dataset = AdditionDataset()
    
    # Initialize model and trainer
    model = PPOAutoregressiveAdditionTransformer(VOCAB_SIZE)
    trainer = PPOTrainer(model, VOCAB_SIZE)
    
    # Train model
    train_metrics = train_ppo_model(model, trainer, dataset)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['rewards'])
    plt.title('Rewards Over Training')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_metrics['policy_losses'])
    plt.title('Policy Losses')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_metrics['value_losses'])
    plt.title('Value Losses')
    
    plt.tight_layout()
    plt.show()
    
    # Model evaluation
    print("\nSample Predictions:")
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            input_str, target_str = dataset.samples[np.random.randint(len(dataset.samples))]
            input_tokens, _ = dataset.tokenize(input_str, target_str)
            
            generated_tokens, _ = model(input_tokens.unsqueeze(0))
            predicted_str = ''.join(map(str, generated_tokens.squeeze().tolist()))
            
            print(f"Input: {input_str}, Target: {target_str}, Predicted: {predicted_str}")

if __name__ == "__main__":
    main()

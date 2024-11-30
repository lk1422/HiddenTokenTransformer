import numpy as np

# Token lookup
TOKEN_LOOKUP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "+": 10, "=": 11, "<PAD>": 12}
REVERSE_LOOKUP = {v: k for k, v in TOKEN_LOOKUP.items()}

def generate_data(num_samples=10000, max_digits=4):
    data = []
    for _ in range(num_samples):
        num1 = np.random.randint(10 ** (max_digits - 1), 10 ** max_digits)
        num2 = np.random.randint(10 ** (max_digits - 1), 10 ** max_digits)
        problem = f"{num1}+{num2}="
        solution = str(num1 + num2)
        data.append((problem, solution))
    return data

# Example usage
train_data = generate_data(num_samples=10000, max_digits=4)
val_data = generate_data(num_samples=1000, max_digits=4)

def encode_sequence(sequence, max_length):
    encoded = [TOKEN_LOOKUP[char] for char in sequence]
    padding_needed = max_length - len(encoded)
    return [TOKEN_LOOKUP["<PAD>"]] * padding_needed + encoded

def preprocess_data(data, max_problem_length, max_solution_length):
    inputs, targets = [], []
    for problem, solution in data:
        inputs.append(encode_sequence(problem, max_problem_length))
        targets.append(encode_sequence(solution, max_solution_length))
    return np.array(inputs), np.array(targets)

# Define sequence lengths
max_problem_length = 3 * 4 + 3  # 4 digits + '+' + '='
max_solution_length = 4         # 4 digits in the solution

# Preprocess training and validation data
X_train, y_train = preprocess_data(train_data, max_problem_length, max_solution_length)
X_val, y_val = preprocess_data(val_data, max_problem_length, max_solution_length)

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=2, num_layers=2, max_seq_len=20, vocab_size=len(TOKEN_LOOKUP)):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=TOKEN_LOOKUP["<PAD>"])
        self.positional_encoding = TransformerFeatureExtractor.create_sinusoidal_embeddings(max_seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, vocab_size)  # Output layer for token predictions

    def forward(self, x):
        padding_mask = x == TOKEN_LOOKUP["<PAD>"]
        embeddings = self.embedding(x.long()) + self.positional_encoding[:, :x.shape[1], :]
        transformer_out = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        return self.fc(transformer_out)

# Instantiate the model
model = TransformerModel(embed_dim=32, num_heads=2, num_layers=2, max_seq_len=max_problem_length)


# Prepare data loaders
train_dataset = TensorDataset(th.tensor(X_train, dtype=th.long), th.tensor(y_train, dtype=th.long))
val_dataset = TensorDataset(th.tensor(X_val, dtype=th.long), th.tensor(y_val, dtype=th.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define optimizer, loss, and device
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_LOOKUP["<PAD>"])

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)  # Shape: (batch_size, seq_len, vocab_size)
        outputs = outputs.view(-1, outputs.size(-1))  # Reshape for loss calculation
        y_batch = y_batch.view(-1)  # Flatten target
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with th.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).view(-1, outputs.size(-1))
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")


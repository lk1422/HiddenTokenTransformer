import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define constants
VOCAB = '0123456789+='
TOKEN2IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
IDX2TOKEN = {idx: ch for ch, idx in TOKEN2IDX.items()}
VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LEN = 10
EMBEDDING_DIM = 32
N_HEADS = 4
NUM_LAYERS = 2
HIDDEN_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 1000

# Create a dataset for learning addition
def generate_data(num_samples=10000):
    data = []
    for _ in range(num_samples):
        a = np.random.randint(0, 1000)
        b = np.random.randint(0, 1000)
        input_str = f"{a}+{b}="
        target_str = str(a + b)
        data.append((input_str, target_str))
    return data

class AdditionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_str, target_str = self.data[idx]
        input_ids = [TOKEN2IDX[ch] for ch in input_str]
        target_ids = [TOKEN2IDX[ch] for ch in target_str]
        input_ids = input_ids + [TOKEN2IDX['=']] * (MAX_SEQ_LEN - len(input_ids))
        target_ids = target_ids + [TOKEN2IDX['=']] * (MAX_SEQ_LEN - len(target_ids))
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

# Define the Transformer model
class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, num_layers, hidden_dim):
        super(AdditionTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, tgt_len):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        logits = self.fc(x)
        return logits

# Training Loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()

            # Predict next tokens in the autoregressive manner
            outputs = model(input_ids, tgt_len=target_ids.size(1))
            outputs = outputs[:, :-1].reshape(-1, VOCAB_SIZE)
            target_ids = target_ids[:, 1:].reshape(-1)

            loss = criterion(outputs, target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Function to generate test outputs autoregressively
def test(model, device, input_str):
    model.eval()
    with torch.no_grad():
        input_ids = [TOKEN2IDX[ch] for ch in input_str]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        generated_ids = input_ids[:]
        for _ in range(MAX_SEQ_LEN - len(input_ids)):
            outputs = model(input_tensor, tgt_len=len(generated_ids))
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            # Append the predicted token to the sequence
            generated_ids.append(next_token_id)

            # Update the input tensor with the new token
            input_tensor = torch.tensor(generated_ids, dtype=torch.long).unsqueeze(0).to(device)

            # Stop if the predicted token is '='
            if IDX2TOKEN[next_token_id] == '=':
                break

        predicted_str = ''.join([IDX2TOKEN[idx] for idx in generated_ids])
        print(f"Input: {input_str}, Predicted Output: {predicted_str}")

# Main function
def main():
    # Prepare dataset and dataloader
    data = generate_data()
    dataset = AdditionDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdditionTransformer(VOCAB_SIZE, EMBEDDING_DIM, N_HEADS, NUM_LAYERS, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, dataloader, optimizer, criterion, device)

    # Test the model
    test_cases = ["12+34=", "56+78=", "123+456=", "789+101="]
    for test_case in test_cases:
        test(model, device, test_case)

if __name__ == "__main__":
    main()

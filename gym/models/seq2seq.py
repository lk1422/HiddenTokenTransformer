import torch
import math
import torch.nn as nn
from token_lookup import TOKEN_LOOKUP

class EncoderDecoderArithmetic(nn.Module):
    def __init__(self, d_model, nhead, n_encoder, n_decoder, d_feedforward, max_seq_len, device):
        super().__init__()
        self.transformer = Seq2Seq(d_model, nhead, n_encoder, n_decoder, d_feedforward, max_seq_len, device)
        self.mlp_head = nn.Sequential(*[
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, len(TOKEN_LOOKUP))
                ])

    def forward(self, obs):
        #src: (S, N, E) tgt: (T, N, E)
        src, tgt = obs['src'], obs['tgt']
        out = self.transformer(src, tgt) #(T, N, E)
        out = self.mlp_head(out) #(T, N, C)
        return out


class Seq2Seq(nn.Module):
    def __init__(self, d_model, nhead, n_encoder, n_decoder, d_feedforward, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.d_feedforward = d_feedforward
        self.device = device

        # Replace learnable positional embeddings with sinusoidal positional encodings
        self.positional_encoding = self.create_sinusoidal_embeddings(max_seq_len, d_model).to(self.device)
        self.embedding = nn.Embedding(len(TOKEN_LOOKUP), d_model, padding_idx=TOKEN_LOOKUP["<PAD>"])
        self.transformer = nn.Transformer(d_model, nhead, n_encoder, n_decoder, d_feedforward)

    @staticmethod
    def create_sinusoidal_embeddings(max_seq_len, embed_dim):
        """
        Create sinusoidal positional embeddings as described in the Transformer paper.
        """
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices in the embedding
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the embedding
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension for compatibility
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)
        return pe

    def generate_square_subsequent_mask(self, sz):
        # Create a square mask for the decoder
        mask = (torch.triu(torch.ones(sz, sz)) != 1).transpose(0, 1)
        return mask

    def forward(self, src, tgt):
        src_padding_mask = (src == TOKEN_LOOKUP["<PAD>"]).to(self.device)
        tgt_padding_mask = (tgt == TOKEN_LOOKUP["<PAD>"]).to(self.device)
        
        # Create causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        
        # Embed and add positional encoding to source and target
        src_embeddings = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_embeddings = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        src_embeddings = src_embeddings.transpose(0, 1)
        tgt_embeddings = tgt_embeddings.transpose(0, 1)



        output = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            src_mask=None,  
            tgt_mask=tgt_mask,  # Causal mask for decoder
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        return output

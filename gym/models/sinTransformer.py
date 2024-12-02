import math
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from token_lookup import TOKEN_LOOKUP

device = th.device("cuda" if th.cuda.is_available() else "cpu")

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=16, num_heads=1, num_layers=1, max_seq_len=20):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)

        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(len(TOKEN_LOOKUP), embed_dim, padding_idx=TOKEN_LOOKUP["<PAD>"])

        # Replace learnable positional embeddings with sinusoidal positional encodings
        self.positional_encoding = self.create_sinusoidal_embeddings(max_seq_len, embed_dim).to(device)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )

        # Final output layer for the latent features
        self.flatten = nn.Flatten()

    @staticmethod
    def create_sinusoidal_embeddings(max_seq_len, embed_dim):
        """
        Create sinusoidal positional embeddings as described in the Transformer paper.
        """
        pe = th.zeros(max_seq_len, embed_dim)
        position = th.arange(0, max_seq_len, dtype=th.float32).unsqueeze(1)
        div_term = th.exp(th.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices in the embedding
        pe[:, 0::2] = th.sin(position * div_term)
        # Apply cosine to odd indices in the embedding
        pe[:, 1::2] = th.cos(position * div_term)
        
        # Add a batch dimension for compatibility
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)
        return pe

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # padded_observations = observations
        padding_mask = observations == TOKEN_LOOKUP["<PAD>"]

        # Embed the observations
        embeddings = self.embedding(observations.long().to(device))
        embeddings += self.positional_encoding[:, :self.max_seq_len, :]

        # Pass through the transformer with an attention mask
        transformer_out = self.transformer(embeddings, src_key_padding_mask=padding_mask)

        # Use the last token's output
        return self.flatten(transformer_out[:, -1, :])  # Flatten the last token's output

# Custom Actor-Critic Policy
class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(embed_dim=16, num_heads=2, num_layers=2, max_seq_len=15 + 6),
            **kwargs,
        )

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from token_lookup import TOKEN_LOOKUP

# device = th.device("cuda" if th.cuda.is_available() else "cpu")
device = th.device("cpu")

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=20):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)

        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(len(TOKEN_LOOKUP), embed_dim, padding_idx=TOKEN_LOOKUP["<PAD>"])

        # Learnable positional encodings
        self.positional_embedding = nn.Parameter(th.zeros(1, max_seq_len, embed_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )

        # Final output layer for the latent features
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # batch_size, seq_len = observations.shape
        #
        # # Compute the number of padding tokens needed
        # padding_needed = self.max_seq_len - seq_len
        # if padding_needed < 0:
        #     raise ValueError("Sequence length exceeds the maximum sequence length!")
        #
        # # Apply front padding with the padding token ("<PAD>")
        # padded_observations = th.cat(
        #     [th.full((batch_size, padding_needed), fill_value=TOKEN_LOOKUP["<PAD>"], dtype=observations.dtype, device=observations.device),
        #      observations],
        #     dim=1
        # )
        #
        # Create padding mask (1 for padding, 0 for actual tokens)
        padded_observations = observations
        padding_mask = padded_observations == TOKEN_LOOKUP["<PAD>"]

        # Embed the padded observations
        embeddings = self.embedding(padded_observations.long())  # Convert tokens to embeddings
        embeddings += self.positional_embedding[:, :self.max_seq_len, :]  # Add positional encodings

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
            features_extractor_kwargs=dict(embed_dim=64, num_heads=2, num_layers=2, max_seq_len=15),
            **kwargs,
        )


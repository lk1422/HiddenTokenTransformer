import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=128, num_heads=4, num_layers=2, max_seq_len=100):
        """
        :param observation_space: (gym.Space) The observation space of the environment.
        :param embed_dim: (int) Embedding dimension for the transformer.
        :param num_heads: (int) Number of attention heads.
        :param num_layers: (int) Number of transformer layers.
        :param max_seq_len: (int) Maximum expected sequence length (used to initialize embeddings).
        """
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)

        self.embedding = nn.Embedding(12, embed_dim)  # Input token embeddings
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)  # Positional embeddings

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )

        # Final output layer for the latent features
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the transformer with dynamic positional encoding.
        :param observations: Input observations (batch, seq_len).
        :return: Latent features for policy and value networks.
        """
        batch_size, seq_len = observations.shape

        # Generate positional indices dynamically
        positions = th.arange(0, seq_len, device=observations.device).unsqueeze(0).expand(batch_size, -1)

        # Token embeddings
        token_embeddings = self.embedding(observations.long())

        # Add positional embeddings dynamically
        position_embeddings = self.positional_embedding(positions)
        embeddings = token_embeddings + position_embeddings

        # Pass through the transformer
        transformer_out = self.transformer(embeddings)

        # Use the last token's output
        return self.flatten(transformer_out[:, -1, :])


# Custom Actor-Critic Policy
class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(embed_dim=64, num_heads=2, num_layers=2, max_seq_len=10),
            **kwargs,
        )



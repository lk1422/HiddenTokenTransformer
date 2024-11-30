import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from token_lookup import TOKEN_LOOKUP

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=20):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)

        self.max_seq_len = max_seq_len

        # Token embeddings
        self.embedding = nn.Embedding(len(TOKEN_LOOKUP), embed_dim, padding_idx=TOKEN_LOOKUP["<PAD>"])

        # Learnable positional encodings
        self.positional_embedding = nn.Parameter(th.zeros(1, max_seq_len, embed_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
        )

        # Shared latent layer (ensures consistent feature dimension)
        self.shared_latent_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("Input observations shape:", observations.shape)

        embeddings = self.embedding(observations.long())
        # print("Embeddings shape:", embeddings.shape)

        embeddings += self.positional_embedding[:, :embeddings.size(1), :]
        transformer_out = self.transformer(embeddings)
        # print("Transformer output shape:", transformer_out.shape)

        last_token_output = transformer_out[:, -1, :]
        # print("Last token output shape:", last_token_output.shape)

        latent_output = self.shared_latent_layer(last_token_output)
        # print("Latent output shape:", latent_output.shape)

        return latent_output


# Custom Actor-Critic Policy with Shared Architecture
class TransformerActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(embed_dim=64, num_heads=4, num_layers=2, max_seq_len=20),
            **kwargs,
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass for policy and value predictions.
        """
        # Use the feature extractor to process observations
        latent_features = self.extract_features(obs)

        # Policy head
        logits = self.action_net(latent_features)
        # print(logits)
        action_distribution = self._get_action_dist_from_latent(latent_features)

        # Choose action
        actions = action_distribution.get_deterministic_actions() if deterministic else action_distribution.sample()

        return actions, self.predict_values(obs), action_distribution.log_prob(actions)

    def predict_values(self, obs):
        """
        Predict values for given observations.
        """
        latent_features = self.extract_features(obs)
        return self.value_net(latent_features)

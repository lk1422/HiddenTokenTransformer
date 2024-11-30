import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, InteractionType
from torch import optim
from torch.distributions import Categorical
from torch.nn import Embedding
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate


class OurEmbedding:
    def __init__(self, device):
        self.token_to_index = {
            "<sos>": 0,
            "<eos>": 1,
            "roses": 2,
            "are": 3,
            "red": 4
        }
        self.vocab_size = len(self.token_to_index)
        self.token_vec_dim = 8
        self.device = device
        self.torch_embedding = Embedding(self.vocab_size, self.token_vec_dim, device=self.device)


class PolicyNetwork(nn.Module):
    def __init__(self, our_embedding, device):
        super().__init__()
        self.our_embedding = our_embedding
        self.device = device
        self.transformer = nn.Transformer(
            d_model=self.our_embedding.token_vec_dim,
            nhead=8 if self.our_embedding.token_vec_dim % 8 == 0 else 2,
            batch_first=True,
            device=self.device
        )
        self.fc = nn.Sequential(
            nn.Linear(self.our_embedding.token_vec_dim, self.our_embedding.token_vec_dim * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.our_embedding.token_vec_dim * 2, self.our_embedding.vocab_size, device=self.device)
        )

    def forward(self, src_token_vecs, target_token_vecs):
        transformer_output = self.transformer(src_token_vecs, target_token_vecs)
        last_token_vec = transformer_output[:, -1, :]
        return self.fc(last_token_vec)


class ValueNetwork(nn.Module):
    def __init__(self, our_embedding, device):
        super().__init__()
        self.our_embedding = our_embedding
        self.device = device
        self.transformer = nn.Transformer(
            d_model=self.our_embedding.token_vec_dim,
            nhead=8 if self.our_embedding.token_vec_dim % 8 == 0 else 2,
            batch_first=True,
            device=self.device
        )
        self.fc = nn.Sequential(
            nn.Linear(self.our_embedding.token_vec_dim, self.our_embedding.token_vec_dim * 2, device=self.device),
            nn.ReLU(),
            nn.Linear(self.our_embedding.token_vec_dim * 2, 1, device=self.device)
        )

    def forward(self, src_token_vecs, generated_token_vecs):
        transformer_output = self.transformer(src_token_vecs, generated_token_vecs)
        last_element = transformer_output[:, -1, :]
        return self.fc(last_element)


import matplotlib.pyplot as plt

def main():
    # Setup
    batch_size = 1
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    our_embedding = OurEmbedding(device)
    policy_network = PolicyNetwork(our_embedding, device)
    value_network = ValueNetwork(our_embedding, device)
    policy_module = TensorDictModule(
        module=policy_network,
        in_keys=["src_token_vecs", "target_token_vecs"],
        out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=True
    )
    value_operator = ValueOperator(
        module=value_network,
        in_keys=["src_token_vecs", "target_token_vecs"],
        out_keys=["value"]
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_operator,
    )
    loss_module.set_keys(
        advantage="advantage",
        value_target="value_target",
        value="value",
        action="action",
        sample_log_prob="sample_log_prob"
    )

    optimizer = optim.Adam(loss_module.parameters())
    max_epoch = 1000
    losses = []
    rewards = []
    advantages = []
    value_targets = []
    loss_critics = []
    loss_entropies = []
    loss_objectives = []

    for epoch in range(max_epoch):
        batch_src_indexes = [[2, 3]] * batch_size
        batch_target_indexes = [[0]] * batch_size

        tensor_dict = TensorDict({}, batch_size=[batch_size])
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)

        next_state_value = torch.zeros(batch_size, 1, device=device) 

        for step in range(10):
            if done_flags.all():
                break

            src_token_vecs = our_embedding.torch_embedding(torch.LongTensor(batch_src_indexes).to(device))
            target_token_vecs = our_embedding.torch_embedding(torch.LongTensor(batch_target_indexes).to(device))

            tensor_dict["src_token_vecs"] = src_token_vecs
            tensor_dict["target_token_vecs"] = target_token_vecs

            value_operator(tensor_dict)  # Compute current state value
            actor(tensor_dict)

            tensor_dict["sample_log_prob"] = tensor_dict["sample_log_prob"].detach()

            next_actions = tensor_dict["action"]
            batch_target_indexes = [
                target + [next_action.item()] for target, next_action in zip(batch_target_indexes, next_actions)
            ]
            done_flags |= next_actions == 1

            # Compute value of the next state
            if step < 9:  # Avoid accessing out-of-bounds values on the last step
                next_src_token_vecs = our_embedding.torch_embedding(torch.LongTensor(batch_src_indexes).to(device))
                next_target_token_vecs = our_embedding.torch_embedding(torch.LongTensor(batch_target_indexes).to(device))
                next_tensor_dict = TensorDict({}, batch_size=[batch_size])
                next_tensor_dict["src_token_vecs"] = next_src_token_vecs
                next_tensor_dict["target_token_vecs"] = next_target_token_vecs
                value_operator(next_tensor_dict)  # Predict next state value
                next_state_value = next_tensor_dict["value"]

        reward = torch.zeros(batch_size, device=device)
        for i, target_tokens in enumerate(batch_target_indexes):
            batch_score = 0
            desired_sequence = [0, 2, 3, 4, 1]
            max_len = len(desired_sequence)

            for j, token in enumerate(target_tokens):
                if j < max_len and token == desired_sequence[j]:
                    batch_score += 0.5
                else:
                    batch_score -= 1

            reward[i] = batch_score

        done = done_flags.clone()
        terminated = done_flags.clone()

        # Compute GAE with the correct `next_state_value`
        advantage, value_target = generalized_advantage_estimate(
            gamma=0.98,
            lmbda=0.95,
            state_value=tensor_dict["value"],
            next_state_value=next_state_value,
            reward=reward.unsqueeze(1),
            done=done.unsqueeze(1),
            terminated=terminated.unsqueeze(1)
        )
        tensor_dict["advantage"] = advantage
        tensor_dict["value_target"] = value_target

        advantages.append(advantage.mean().item())  # Track mean advantage
        value_targets.append(value_target.mean().item())  # Track mean value target

        loss_tensor_dict = loss_module(tensor_dict)
        loss_critic = loss_tensor_dict["loss_critic"]
        loss_entropy = loss_tensor_dict["loss_entropy"]
        loss_objective = loss_tensor_dict["loss_objective"]
        loss = loss_critic + loss_entropy + loss_objective

        loss_critics.append(loss_critic.item())  # Track loss components
        loss_entropies.append(loss_entropy.item())
        loss_objectives.append(loss_objective.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())  # Track total loss
        print(f"Epoch: {epoch}, Total Loss: {loss.item()}")

    # Plot metrics in a single figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()

    axes[0].plot(losses, label="Total Loss")
    axes[0].set_title("Total Loss Over Time")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(rewards, label="Mean Reward")
    axes[1].set_title("Mean Reward Over Time")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reward")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(advantages, label="Mean Advantage")
    axes[2].set_title("Mean Advantage Over Time")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Advantage")
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(value_targets, label="Mean Value Target")
    axes[3].set_title("Mean Value Target Over Time")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Value Target")
    axes[3].legend()
    axes[3].grid()

    axes[4].plot(loss_critics, label="Loss Critic")
    axes[4].plot(loss_entropies, label="Loss Entropy")
    axes[4].plot(loss_objectives, label="Loss Objective")
    axes[4].set_title("Loss Components Over Time")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("Loss")
    axes[4].legend()
    axes[4].grid()

    # Hide the last subplot if not needed
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save({
        'policy_network': policy_network.state_dict(),
        'value_network': value_network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, "ppo_model.pt")
    print("Model saved as ppo_model.pt")


if __name__ == "__main__":
    main()

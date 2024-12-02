import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from arithmetic_gyms import additionEOSHiddenST as cur_gym
from models import seq2seq as cur_transformer


def train_model(config_idx, num_digits, policy_kwargs, use_hidden):
    """Function to train a model with a specific configuration."""
    # Define unique save paths for each configuration
    log_dir = f"./logs/hidden_{use_hidden}_config_{config_idx}/"
    os.makedirs(log_dir, exist_ok=True)

    # Save a checkpoint every 50,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Create the environment
    env = cur_gym.TextGym(max_digits=num_digits, use_hidden=use_hidden)

    # Instantiate and configure the PPO model
    model = PPO(
        cur_transformer.TransformerActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
    )

    # Train the model
    model.learn(
        total_timesteps=500_000_000,
        callback=checkpoint_callback,
    )


def main():
    # Set the number of digits for the arithmetic task
    num_digits = 4

    # Define the transformer policy parameters
    params = [
        dict(embed_dim=8, d_feedforward=32, num_heads=1, num_layers=1),
        dict(embed_dim=16, d_feedforward=64, num_heads=1, num_layers=1),
        dict(embed_dim=16, d_feedforward=64, num_heads=1, num_layers=4),
        dict(embed_dim=16, d_feedforward=64, num_heads=2, num_layers=1),
        dict(embed_dim=32, d_feedforward=128, num_heads=4, num_layers=1),
        dict(embed_dim=32, d_feedforward=128, num_heads=4, num_layers=2),
        dict(embed_dim=128, d_feedforward=256, num_heads=4, num_layers=4),
        dict(embed_dim=128, d_feedforward=512, num_heads=4, num_layers=2),
    ]

    # Create a combined list of parameter sets with `use_hidden` variations
    combined_configs = []
    for param in params:
        combined_configs.append((param, True))
        combined_configs.append((param, False))

    # Spawn a process for each configuration
    processes = []
    for idx, (policy_kwargs, use_hidden) in enumerate(combined_configs):
        p = multiprocessing.Process(
            target=train_model,
            args=(idx, num_digits, policy_kwargs, use_hidden)
        )
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()

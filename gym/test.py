from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

# from arithmetic_gyms import additionEOSHidden as cur_gym
from arithmetic_gyms import additionEOSHiddenST as cur_gym
# from models import sinTransformer as cur_transformer
from models import seq2seq as cur_transformer
from main import load
import torch

device = torch.device('cuda')

def load_and_test_model():
    # Load the saved model
    model_path = "./logs/rl_model_4000000_steps.zip"  # Update with the correct path to the saved model
    num_digits = 4  # Same number of digits used during training

    # Create the environment
    env = cur_gym.TextGym(max_digits=num_digits)

    # Load the model
    model = PPO.load(model_path, custom_objects={
        'policy': cur_transformer.TransformerActorCriticPolicy
    })

    print("Model loaded successfully.")

    # Test the model
    total_episodes = 10  # Number of episodes to test
    total_reward = 0  # Track cumulative rewards across all episodes

    for episode in range(total_episodes):
        obs = env.reset()
        episode_reward = 0  # Track reward for the current episode
        done = False

        while not done:
            action, _ = model.predict(obs)  # Predict the next action
            obs, reward, done, _ = env.step(int(action))  # Take a step
            episode_reward += reward  # Accumulate rewards

            # Decode the predicted action
            predicted_char = str(action)
            print(f"Predicted: {predicted_char}, Reward: {reward}")

            # Optional: Render the environment
            # env.render()

        env.render()
        total_reward += episode_reward  # Add episode reward to total reward
        print(f"Episode {episode + 1} Reward: {episode_reward}")
        print("-" * 50)

    print(f"Average Reward over {total_episodes} episodes: {total_reward / total_episodes}")

def test_model():
    num_digits = 4  # Same number of digits used during training

    # Create the environment
    env = cur_gym.TextGym(max_digits=num_digits)

    # Load the model
    num_digits = 4  # Same number of digits used during training

    # Create the environment
    env = cur_gym.TextGym(max_digits=num_digits)

    # Load the model
    model = PPO(
        cur_transformer.TransformerActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="./ppo_addition_logs/",
        learning_rate=3e-4,
        ent_coef=0.02,
        device=device,
        policy_kwargs={
            "d_model": 128,
            "nhead"  : 4,
            "n_encoder" : 2,
            "n_decoder" : 2,
            "d_feedforward" : 256,
            "net_arch": dict(pi=[128]),
            "device": device,
            "share_features_extractor": True
            },
    )
    load(model, "../Supervised/params/model_parameters_124999.pth")


    print("Model loaded successfully.")

    # Test the model
    total_episodes = 10  # Number of episodes to test
    total_reward = 0  # Track cumulative rewards across all episodes

    for episode in range(total_episodes):
        obs = env.reset()
        episode_reward = 0  # Track reward for the current episode
        done = False

        while not done:
            action, _ = model.predict(obs)  # Predict the next action
            obs, reward, done, _ = env.step(int(action))  # Take a step
            episode_reward += reward  # Accumulate rewards

            # Decode the predicted action
            predicted_char = str(action)
            print(f"Predicted: {predicted_char}, Reward: {reward}")

            # Optional: Render the environment
            # env.render()

        env.render()
        total_reward += episode_reward  # Add episode reward to total reward
        print(f"Episode {episode + 1} Reward: {episode_reward}")
        print("-" * 50)

    print(f"Average Reward over {total_episodes} episodes: {total_reward / total_episodes}")

if __name__ == '__main__':
    #load_and_test_model()
    test_model()

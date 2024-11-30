from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from arithmetic_gyms import multiplication as cur_gym
from models import paddedTransformer as cur_transformer

from stable_baselines3.common.callbacks import CheckpointCallback

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

num_digits = 4
# Create the environment
env = cur_gym.TextGym(max_digits=num_digits)

# policy_kwargs = dict(net_arch=dict(pi=[256], vf=[256]))
# policy_kwargs = {}

# Define and train the PPO model
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_addition_logs/")
model = PPO(
    cur_transformer.TransformerActorCriticPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_addition_logs/",
    learning_rate=3e-4,
    # policy_kwargs={"seq_len": 16},
    batch_size=32,
)

model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback,
)

# Test the model
total_reward = 0  # Track total reward

obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs)  # Predict the next action
    obs, reward, done, _ = env.step(int(action))  # Take a step
    total_reward += reward  # Accumulate rewards
    
    # Decode the predicted action
    predicted_char = str(action) if action < 10 else "="

    # env.render()  # Render the environment
    
    if done:
        # Print cumulative predictions and total reward
        env.render()  # Render the environment
        print("Total Reward for this Problem:", total_reward)
        print()

        # Reset for the next problem
        cumulative_predictions = []
        total_reward = 0
        obs = env.reset()

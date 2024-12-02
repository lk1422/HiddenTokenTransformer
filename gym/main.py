import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

# from arithmetic_gyms import multiplication as cur_gym
from arithmetic_gyms import additionEOSHiddenST as cur_gym
# from arithmetic_gyms import additionEOSHidden as cur_gym
# from arithmetic_gyms import additionEOS as cur_gym
# from arithmetic_gyms import addition as cur_gym
# from models import paddedTransformer as cur_transformer
from models import seq2seq as cur_transformer
# from models import sinTransformer as cur_transformer

from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
      save_freq=50000,
      save_path="./logs/",
      name_prefix="rl_model",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )

    num_digits = 4

    vectorize = False
    num_envs = 4
    # Create the environment
    env = cur_gym.TextGym(max_digits=num_digits, use_hidden=True)

    def make_env(seed):
        def _init():
            env = cur_gym.TextGym(max_digits=num_digits)
            return env
        return _init

    if vectorize:
        env = make_vec_env(env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
        # env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    # policy_kwargs = dict(net_arch=dict(pi=[256], vf=[256]))
    # policy_kwargs = {}

    # Define and train the PPO model
    # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_addition_logs/")
    # model = PPO(
    #     cur_transformer.TransformerActorCriticPolicy,
    #     # "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="./ppo_addition_logs/",
    #     learning_rate=3e-4,
    #     ent_coef=0.02,
    #     # gamma=.99,
    #     #batch_size=2
    #     # learning_rate=1e-3,
    #     # policy_kwargs={"seq_len": 16},
    # )
    from stable_baselines3.common.callbacks import CallbackList, BaseCallback

    class CustomLoggerCallback(BaseCallback):
        """
        Example of a custom callback to log values to TensorBoard.
        """
        def __init__(self, verbose=0):
            super(CustomLoggerCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log a custom value (replace with your logic)
            infos = self.locals.get('infos', [])
            if 'terminal_observation' in infos[0]:
                if len(infos) > 1:
                    print('error!!!')
                self.logger.record("custom/metric", infos[0]["digits_correct"])
            return True

    custom_logger_callback = CustomLoggerCallback()
    callback_list = CallbackList([custom_logger_callback, checkpoint_callback])

    model_path = "./logs/rl_model_2600000_steps.zip"  # Update with the correct path to the saved model
    model = PPO.load(model_path, env, custom_objects={
        'policy': cur_transformer.TransformerActorCriticPolicy,
        'tensorboard_log':"./ppo_addition_logs/",
        'learning_rate':3e-4,
        'ent_coef':0.02,
        'gamma':1,
        # 'gae_lambda': .90
    })

    model.learn(
        total_timesteps=500_000_000,
        # callback=checkpoint_callback,
        callback=callback_list,
    )

    # Test the model
    total_reward = 0  # Track total reward

    obs = env.reset()
    for _ in range(20):
        action, _ = model.predict(obs)  # Predict the next action
        print(action)
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

if __name__ == '__main__':
    main()

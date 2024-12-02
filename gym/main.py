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

device = torch.device('cuda')

def load(model, param_file):
    #Make Copies and then put them into model withe new names
    params = torch.load(param_file)
    new_state_dict ={}
    for key, value in params.items():
        new_state_dict["features_extractor." + str(key)] =  value
        #new_state_dict["pi_features_extractor" + key] = value
        #features_extractor.transformer.embedding.weight'

    model.set_parameters(dict(policy=new_state_dict), exact_match=False)


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
    model = PPO(
        cur_transformer.TransformerActorCriticPolicy,
        # "MlpPolicy",
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
            "device": device, 
            "share_features_extractor":True
            },
    )
    example_weight = "features_extractor.transformer.transformer.encoder.layers.0.self_attn.out_proj.weight"

    print("BEFORE")
    print(model.get_parameters()["policy"][example_weight])
    load(model, "../Supervised/params/model_parameters_109999.pth")
    print("AFTER")
    print(model.get_parameters()["policy"][example_weight])

    # model_path = "./logs/rl_model_50000_steps.zip"  # Update with the correct path to the saved model
    # model = PPO.load(model_path, env, custom_objects={
    #     'policy': cur_transformer.TransformerActorCriticPolicy,
    #     'tensorboard_log':"./ppo_addition_logs/",
    #     'learning_rate':3e-4,
    #     'ent_coef':0.03,
    #     'gamma':.90,
    #     'gae_lambda': .90
    # })

    model.learn(
        total_timesteps=500_000_000,
        callback=checkpoint_callback,
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

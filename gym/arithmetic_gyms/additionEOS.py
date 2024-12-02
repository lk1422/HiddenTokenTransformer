import gym
from gym import spaces
import numpy as np
from token_lookup import TOKEN_LOOKUP, REVERSE_LOOKUP
import torch as th

device = th.device("cpu")

class TextGym(gym.Env):
    def __init__(self, max_digits=2, eos_penalty=10):
        super(TextGym, self).__init__()
        self.max_digits = max_digits
        self.eos_penalty = eos_penalty  # Penalty for not using EOS correctly

        # Observation space: tokenized problem as a sequence
        self.observation_space = spaces.Box(
            low=0, high=max(TOKEN_LOOKUP.values()), shape=(self.max_digits * 3 + 3,), dtype=np.int32
        )

        # Action space: predict a token (digits 0-9, '+', '=', '<PAD>', '<EOS>')
        self.action_space = spaces.Discrete(len(TOKEN_LOOKUP))

        self.current_problem = None
        self.solution = None
        self.current_index = 0
        self.state = None
        self.predictions = []  # Store predictions dynamically
        self.extra_token_count = 0  # Track extra tokens beyond EOS

    def _generate_problem(self):
        """Generate a single random problem with digits up to max_digits in length."""
        num1 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        num2 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        problem = f"{num1}+{num2}="
        solution = str(num1 + num2)
        return problem, solution

    def _encode_problem(self, problem):
        """Encode a problem string into a numerical sequence."""
        encoded = [TOKEN_LOOKUP[char] for char in problem]
        return np.array(encoded, dtype=np.int32)

    def reset(self):
        """Reset the environment to a new problem."""
        self.current_problem, self.solution = self._generate_problem()
        self.current_index = 0
        self.state = self._encode_problem(self.current_problem).tolist()
        self.predictions = []  # Clear predictions
        self.extra_token_count = 0  # Reset extra token counter

        # Add front padding to match the observation space shape
        padding_needed = self.observation_space.shape[0] - len(self.state)
        padded_state = [TOKEN_LOOKUP["<PAD>"]] * padding_needed + self.state
        return th.tensor(padded_state, dtype=th.int32, device=device)

    def step(self, action):
        """Take a step in the environment."""
        done = False
        reward = 0

        # Decode action into a character
        predicted_char = REVERSE_LOOKUP[action]
        remaining_chars = len(self.solution) - self.current_index

        # Append the prediction to the state
        self.predictions.append(action)
        self.state.append(action)

        if predicted_char == "<EOS>":
            if self.current_index == len(self.solution):
                reward = 1
                done = True
            else:
                reward = -remaining_chars
                done = True

        elif self.current_index < len(self.solution):
            # Reward based on correctness of the solution tokens
            correct_char = self.solution[self.current_index]
            if predicted_char == correct_char:
                reward = 1  # Correct prediction
            else:
                reward = -1  # Incorrect prediction
        else:
            # Penalize extra tokens after EOS or solution completion
            if predicted_char == "<EOS>":
                reward = 0
                done = True
            else:
                reward = -1

        # Penalty for failing to predict <EOS>
        if self.current_index >= len(self.solution) and not done and predicted_char != "<EOS>":
            reward = -1
            # done = True

        padding_needed = self.observation_space.shape[0] - len(self.state)

        if padding_needed == 0:
            reward = -2
            done = True

        self.current_index += 1

        # Add front padding to match the observation space shape
        padding_needed = self.observation_space.shape[0] - len(self.state)
        padded_state = [TOKEN_LOOKUP["<PAD>"]] * padding_needed + self.state
        return (
            th.tensor(padded_state, dtype=th.int32, device=device),
            th.tensor(reward, dtype=th.float32, device=device),
            done,
            {},
        )

    def render(self, mode="human"):
        """Render the current problem and progress."""
        progress = "".join(REVERSE_LOOKUP[token] for token in self.predictions)
        print(f"Problem: {self.current_problem}, Predicted Output: {progress}")

import gym
from gym import spaces
import numpy as np
from token_lookup import TOKEN_LOOKUP, REVERSE_LOOKUP
import torch as th

device = th.device("cpu")

class TextGym(gym.Env):
    def __init__(self, max_digits=4, eos_penalty=10, use_hidden=True, buffer_factor=2, max_hidden_tokens=5):
        super(TextGym, self).__init__()
        self.max_digits = max_digits
        self.eos_penalty = eos_penalty
        self.use_hidden = use_hidden

        # Observation space: separate src and target
        self.src_length = self.max_digits * 2 + 2  # e.g., "12+34="
        self.tgt_length =  self.max_digits + (buffer_factor * max_hidden_tokens) + 1
        self.observation_space = spaces.Dict({
            "src": spaces.Box(
                low=0, high=max(TOKEN_LOOKUP.values()), shape=(self.src_length,), dtype=np.int32
            ),
            "tgt": spaces.Box(
                low=0, high=max(TOKEN_LOOKUP.values()), shape=(self.tgt_length,), dtype=np.int32
            ),
            "step": spaces.Box(
                low=0, high=np.inf, shape=(), dtype=np.int32
            )
        })

        self.action_space = spaces.Discrete(len(TOKEN_LOOKUP))

    def _generate_problem(self):
        """Generate a single random problem with digits up to max_digits in length."""
        num1 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        num2 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        problem = f"{num1}*{num2}="
        solution = str(num1 * num2)

        # print(solution)
        encoded_padded = self._encode_problem(problem, self.src_length)

        return encoded_padded, solution

    def _encode_problem(self, problem, pad_to):
        """Encode a problem string into a numerical sequence."""
        encoded = [TOKEN_LOOKUP[char] for char in problem]
        encoded = np.array(encoded, dtype=np.int32)

        padded = np.pad(
            encoded, (0, pad_to - len(encoded)), constant_values=TOKEN_LOOKUP["<PAD>"]
        )

        return padded

    def reset(self):
        self.cur_step = 0
        self.src, self.solution = self._generate_problem()
        self.full_state = ["<S>"]
        self.visible_predictions = []
        self.current_index = -1
        self.has_started = False

        tgt = self._encode_problem(self.full_state, self.tgt_length)
        return {
            "src": th.tensor(self.src, dtype=th.int32, device=device),
            "tgt": th.tensor(tgt, dtype=th.int32, device=device),
            "step": 0
        }

    def step(self, action, missing_char_factor=2.0):
        self.cur_step += 1
        done = False
        reward = 0

        predicted_char = REVERSE_LOOKUP[action]

        self.full_state.append(predicted_char)
        if self.has_started or (not self.use_hidden):
            self.visible_predictions.append(predicted_char)
            self.current_index += 1

        remaining_chars = max(0, len(self.solution) - self.current_index)

        is_success = False
        num_correct = 0

        if predicted_char == "<EOS>":
            preds = "".join(str(a) for a in self.visible_predictions[:-1]) 
            is_success = preds == self.solution

            num_correct = sum(1 if a == b else 0 for a, b in zip(preds, self.solution))
            bonus = 2 * is_success

            reward = -(remaining_chars) + (self.has_started and self.use_hidden) + bonus
            # reward = -(remaining_chars) + (self.has_started and self.use_hidden) + 
            done = True
        elif predicted_char == "<H>":
            if (not self.has_started) and self.use_hidden:
                self.has_started = True
            else:
                reward = -1 # Wrong char
        elif self.has_started or (not self.use_hidden):
            if self.current_index < len(self.solution):
                correct_char = self.solution[self.current_index]
                if predicted_char == correct_char:
                    reward = 1
                else:
                    reward = 0 # No reward for wrong char
            else:
                reward = -1 # Punishment for too many chars
        else:
            reward = 0 # No reward for hidden token

        # Return src (unchanged) and padded predictions as tgt
        src = self.src
        if len(self.full_state) > self.tgt_length:
            reward = -missing_char_factor * (remaining_chars + 1)
            done = True
        tgt = self._encode_problem(self.full_state[:self.tgt_length], pad_to=self.tgt_length)

        return {
            "src": th.tensor(src, dtype=th.int32, device=device),
            "tgt": th.tensor(tgt, dtype=th.int32, device=device),
            "step": th.tensor(self.cur_step, dtype=th.int32, device=device),
        }, th.tensor(reward, dtype=th.float32, device=device), done, {"is_success": is_success, "digits_correct": num_correct}

    def render(self, mode="human"):
        # progress = "".join([token] for token in self.full_state)

        print(f"Problem: {self.src}, Predicted Output: {self.full_state}")


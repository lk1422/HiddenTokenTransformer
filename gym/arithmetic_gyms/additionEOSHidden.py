import gym
from gym import spaces
import numpy as np
from token_lookup import TOKEN_LOOKUP, REVERSE_LOOKUP
import torch as th

device = th.device("cpu")

class TextGym(gym.Env):
    def __init__(self, max_digits=4, eos_penalty=10, use_hidden=True):
        super(TextGym, self).__init__()
        self.max_digits = max_digits
        self.eos_penalty = eos_penalty  # Penalty for not using EOS correctly
        self.use_hidden = use_hidden

        # Observation space: tokenized problem as a sequence
        self.observation_space = spaces.Box(
            low=0, high=max(TOKEN_LOOKUP.values()), shape=(self.max_digits * 3 + 3 + 6,), dtype=np.int32
        )

        # Action space: predict a token (digits 0-9, '+', '=', '<PAD>', '<EOS>')
        self.action_space = spaces.Discrete(len(TOKEN_LOOKUP))

        self.current_problem = None
        self.solution = None
        self.current_index = -1
        self.state = None
        self.predictions = []  # Store predictions dynamically
        self.extra_token_count = 0  # Track extra tokens beyond EOS
        self.has_started = False  # Mark the start of the "thinking" phase


    def _generate_problem(self):
        """Generate a single random problem with digits up to max_digits in length."""
        num1 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        num2 = np.random.randint(1, 10 ** self.max_digits)  # Up to max_digits long
        problem = f"{num1}+{num2}="
        solution = str(num1 + num2)
        # print(solution)
        return problem, solution

    def _encode_problem(self, problem):
        """Encode a problem string into a numerical sequence."""
        encoded = [TOKEN_LOOKUP[char] for char in problem]
        return np.array(encoded, dtype=np.int32)

    def reset(self):
        """Reset the environment to a new problem."""
        self.current_problem, self.solution = self._generate_problem()
        self.current_index = -1
        self.state = self._encode_problem(self.current_problem).tolist()
        self.predictions = []  # Clear predictions
        self.extra_token_count = 0  # Reset extra token counter
        self.has_started = False  # Mark the start of the "thinking" phase

        # Add front padding to match the observation space shape
        padding_needed = self.observation_space.shape[0] - len(self.state)
        padded_state = [TOKEN_LOOKUP["<PAD>"]] * padding_needed + self.state
        return th.tensor(padded_state, dtype=th.int32, device=device)

    def step(self, action, missing_char_factor=2.0):
        """Take a step in the environment."""
        done = False
        reward = 0

        # Decode action into a character
        predicted_char = REVERSE_LOOKUP[action]

        # Append the prediction to the state
        if self.has_started or (not self.use_hidden):
            self.predictions.append(action)
            self.current_index += 1

        remaining_chars = max(0, len(self.solution) - self.current_index)
        # print("remaining chars")
        # print(remaining_chars)
        # print(self.solution)

        # print(f"current index {self.current_index}")
        self.state.append(action)

        # print("started")
        # print(self.use_hidden)
        # print("use_hidden")
        # print(self.use_hidden)
        is_success = False

        if predicted_char == "<EOS>":
            # print(self.predictions , self.solution)
            is_success = "".join(str(a) for a in self.predictions[:-1]) == self.solution
            reward = - (missing_char_factor *remaining_chars) + (self.has_started and self.use_hidden)
            done = True

        elif predicted_char == "<H>":
            if (not self.has_started) and self.use_hidden:
                self.has_started = True  # Mark the start of the "thinking" phase
                reward = 0  # No immediate reward for predicting <H>
            else:
                reward = -1  # Penalize multiple <H> tokens if invalid


        elif self.has_started or (not self.use_hidden):
            # Reward or penalize based on correctness after <H>
            if self.current_index < len(self.solution):
                correct_char = self.solution[self.current_index]
                if predicted_char == correct_char:
                    reward = 1  # Correct prediction
                else:
                    reward = -1  # Incorrect prediction
                    # print("wanted ")
                    # print(self.solution[self.current_index])
            else:
                # Penalize extra tokens after the solution completion
                reward = -1

        else:
            # Little to no reward for tokens before <H>
            reward = 0


        padding_needed = self.observation_space.shape[0] - len(self.state)

        if padding_needed < 0:
            reward = -missing_char_factor * (remaining_chars + 1)
            done = True


        # Add front padding to match the observation space shape
        padding_needed = self.observation_space.shape[0] - len(self.state)
        padded_state = [TOKEN_LOOKUP["<PAD>"]] * padding_needed + self.state

        # reward = is_success

        return (
            th.tensor(padded_state, dtype=th.int32, device=device),
            th.tensor(reward, dtype=th.float32, device=device),
            done,
            {"is_success": is_success},
        )

    def render(self, mode="human"):
        """Render the current problem and progress."""
        progress = "".join(REVERSE_LOOKUP[token] for token in self.predictions)
        full_state = "".join(REVERSE_LOOKUP[token] for token in self.state)
        print(f"Problem: {self.current_problem}, Predicted Output: {progress}")
        print(f"Total State: {full_state}")


def main():
    # Initialize the environment
    env = TextGym(max_digits=4)
    
    # Patch the global variables for the environment to use
    globals().update({"TOKEN_LOOKUP": TOKEN_LOOKUP, "REVERSE_LOOKUP": REVERSE_LOOKUP})
    
    # Reset the environment
    state = env.reset()
    done = False
    cumulative_reward = 0  # Initialize cumulative reward
    
    print("\nWelcome to the TextGym game!")
    print("You will be given a math problem (e.g., 12+34=).")
    print("Your goal is to predict the digits of the solution, and use <EOS> when done.")
    print("Start by typing <H> to begin the guessing phase.\n")
    print("Actions: Digits 0-9, <H>, <EOS>")
    
    while not done:
        env.render()
        print("Current State:", state.cpu().numpy())

        # Take input from the user
        user_input = input("Enter your action (0-9, +, =, <H>, <EOS>): ").strip()
        if user_input not in TOKEN_LOOKUP:
            print("Invalid action. Please try again.")
            continue

        # Map the input to an action
        action = TOKEN_LOOKUP[user_input]

        # Step through the environment
        state, reward, done, info = env.step(action)
        cumulative_reward += reward.item()  # Update cumulative reward

        # Provide feedback
        print(f"Reward: {reward.item()}")
        print(f"info: {info}")
        print(f"Cumulative Reward: {cumulative_reward}\n")
        if done:
            print("Game over!")
            env.render()

if __name__ == "__main__":
    main()

import numpy as np
from .utils import softmax

LEFT  = 0
RIGHT = 1
UP    = 2
DOWN  = 3

class GridWorld():
    """
    Slightly More complicated RL Enviorment.
    To test the ability of PPO of converging upon more
    complex policy.
    """
    def __init__(self, gw_file):
        """
        Initialize GridWorld object

        Parameters:
            gw_file: file specifiying the format of the gw

            step_reward,fail_reward,destination_reward,probability_failure
            step_reward: reward at each time step (should be negative to
                         push the agent toward completing the maze)
            fail_reward: reward for episode terminating without reaching the destination
                         (agent goes into a terminal location).
            destination_reward: reward for reaching destination
            probability_failure: the probability of a given action not being preformed
                                 ie P(UP| UP) = (1-probability_failure), the rest of the
                                 possible actions are equally as likely (failure / 3).

            This is followed by a representation of the grid world

            S..........
            XXXX.......
            XX....X....
            D..XX......

            S: Start
            D: Destination
            X: Terminal
            .: Regular

        Class Members:
            self.terminal binary vector, where self.terminal[state] <==> state is terminal.
            self.destination binary vector wehere self.destintation[state] <==> state is a destination
            self.start is a scalar representing which state is the starting state (can only be one)
            self.rows, self.cols: # of ros and cols

        """
        self.terminal           = -1
        self.destination        = -1
        self.start              = -1
        self.rows               = -1
        self.cols               = -1
        self.states             = -1
        self.step_reward        = -1
        self.terminal_reward    = -1
        self.destination_reward = -1
        self.p_failure          = -1

        self.SOS                = -1
        self.PAD                = -1
        self.n_tokens           = -1

        self._read_file(gw_file)

    def _build_action_probs(self):
        """
        self._action_probs[i] = PR(action | requested=i)
        """
        self._action_probs = np.zeros((4, 4))
        for i in range(4):
            self._action_probs[i][i] = 1-self.p_failure
            for j in range(4):
                if i == j: continue
                self._action_probs[i][j] = self.p_failure/3

    def _handle_parameters(self, parameters):
        self.step_reward = float(parameters[0])
        self.terminal_reward = float(parameters[1])
        self.destination_reward = float(parameters[2])
        self.p_failure = float(parameters[3])

    def _handle_grid(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.states = self.rows * self.cols
        self.SOS = self.states+1
        self.PAD = self.states+2
        self.n_tokens = self.states+3
        self.terminal = np.zeros(self.rows*self.cols).astype(np.int32)
        self.destination = np.zeros(self.rows*self.cols).astype(np.int32)
        for i in range(self.rows):
            assert len(grid[i]) == self.cols
            for j in range(self.cols):
                if grid[i][j] == "S":
                    self.start = i*self.cols + j
                elif grid[i][j] == "X":
                    self.terminal[i*self.cols+j] = 1
                elif grid[i][j] == "D":
                    self.destination[i*self.cols+j] = 1

        self._build_action_probs()

    def _read_file(self, gw_file_path):
        with open(gw_file_path, 'r')  as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            assert len(lines) > 1

            parameters = lines[0].split(",")
            self._handle_parameters(parameters)
            self._handle_grid(lines[1:])


    def get_start_state(self, n):
        return (np.ones((n,1)) * self.start).astype(np.int32)

    def get_non_terminal(self):
        terminal = (self.terminal | self.destination)
        return np.arange(0, self.states)[np.logical_not(terminal)]

    def preform_action(self, state, actions):
        """
        State: current state
        action: action to be preformed.
        """
        state = state.reshape(-1)

        is_up    = (actions==UP)
        is_down  = (actions==DOWN)
        is_left  = (actions==LEFT)
        is_right = (actions==RIGHT)

        action_up = state - self.cols
        action_down = state + self.cols
        action_left = state - 1
        action_right = state + 1
        
        preformed_action = (is_up * action_up) + (is_down * action_down) + \
                (is_left * action_left) + (is_right * action_right)
        preformed_action = np.clip(preformed_action, 0, self.states-1)

        terminal = (self.terminal[state] | self.destination[state])

        return (terminal * state) + \
                (np.logical_not(terminal) * preformed_action)

    def batched_step(self, state, action):
        """
        State: Current State (N, 1)
        Action: Action requested by agent. (N,)
        """
        N = state.shape[0]

        action_preformed = np.array([np.random.choice(a=4, p=self._action_probs[action[i]]) \
                       for i in range(N)]).reshape(-1)

        preform_action = self.preform_action(state, action_preformed).astype(np.int32)

        terminal = (self.terminal[preform_action] | self.destination[preform_action])
        terminal = terminal.astype(np.int32)

        terminal_rewards = (self.terminal[preform_action] * self.terminal_reward + \
                            self.destination[preform_action] * self.destination_reward)

        rewards = np.logical_not(terminal) * self.step_reward + terminal_rewards


        return preform_action, rewards, terminal

class basicMDP():
    """
    Basic RL Enviornment for Debugging RL Algs
    It is a 5 state MDP with model dynamics 
    hard coded below.
    """
    def __init__(self):
        self.states = 6
        self.rewards = np.zeros((6, 6))
        self.pr = np.zeros((6,6,2))
        self.terminal = set([1, 2, 4, 6])
        self._init_rewards()
        self._init_transition_probs()

    def _init_rewards(self):
        """
        Describes all non-zero rewards from transition of
        state1 to state2.
        """
        self.rewards[0, 1] =  1/50
        self.rewards[0, 2] = -5/50
        self.rewards[3, 2] = -5/50
        self.rewards[3, 4] = 1

    def _init_transition_probs(self):
        """
        Describes the transition probabilities
        self.pr[s_t, s_prev, a_prev] = pr(s_t | s_prev, a_prev)
        a_prev = {Left(0), Right(1)}
        """
        self.pr[1, 0, RIGHT] = 1
        self.pr[3, 0,  LEFT] = 0.5
        self.pr[2, 0,  LEFT] = 0.5

        self.pr[4, 3,  LEFT] = 0.5
        self.pr[2, 3,  LEFT] = 0.5
        self.pr[0, 3, RIGHT] = 0.5
        self.pr[2, 3, RIGHT] = 0.5

        self.pr[1,1,  LEFT] = 1
        self.pr[1,1, RIGHT] = 1
        self.pr[2,2,  LEFT] = 1
        self.pr[2,2, RIGHT] = 1
        self.pr[4,4,  LEFT] = 1
        self.pr[4,4, RIGHT] = 1
        self.pr[5,5,  LEFT] = 1
        self.pr[5,5, RIGHT] = 1

    def get_start_state(self, n):
        start =  np.zeros((n, 1, 6))
        start[:, :, 0] = 1
        return start

    def batched_step(self, state, action):
        """
        state : (N, 6) current_state array
        action: (N,) action array 
        Returns (next_states, rewards, Terminal)
        """
        N = state.shape[0]

        state = np.argmax(state, axis=1)

        transition_probs = self.pr[:, state, action].T

        next_state = np.array([np.random.choice(a=6, p=transition_probs[i]) \
                       for i in range(N)]).reshape(-1)

        next_state_binary = np.zeros((N, 6))
        next_state_binary[np.arange(0,N), next_state] = 1

        rewards = self.rewards[state, next_state]
        terminal = (next_state == 1) | (next_state == 2) | (next_state == 4) | (next_state == 6)

        return next_state_binary, rewards, terminal


from typing import Optional
import numpy as np
import gymnasium as gym

# 4x4 gridworld based on section 4.1 of http://incompleteideas.net/book/RLbook2020.pdf
class GridWorld(gym.Env):
    def __init__(self, size: int=4):
        # size of the square grid (4x4 by default)
        self.size = size

        # initialize positions to uninitialized state
        self._agent_location = np.array([-1, -1], dtype=np.int32)

        # set predefined terminal states
        self._terminal_states = [
            np.array([0,0], dtype=np.int32),
            np.array([size-1, size-1], dtype=np.int32),
        ]

        # define what the agent can observe - self._agent_location
        self.observation_space = gym.spaces.Box(0, size-1, shape=(2,), dtype=int)

        # define what actions are available (4 directions: up, down, left, right)
        self.action_space = gym.spaces.Discrete(4)

        # map action numbers to actual movements on the grid
        # [row, col]
        self._action_to_direction = {
            0: np.array([0, 1]), # move right
            1: np.array([-1, 0]), # move up
            2: np.array([0, -1]), # left
            3: np.array([1, 0]), # down
        }

    def _get_obs(self) -> np.array:
        """
            convert states to available observations

            Returns:
                array: Observation with agent position
        """
        return self._agent_location
    
    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new episode.

            Args:
                seed: Random seed for reproducible episodes
                options: Additional configuration (unused in this example)

            Returns:
                tuple: (observation, info) for the initial state
        """

        # seed the random number generator
        super().reset(seed=seed)

        while True:
            # randomly place agent [x, y] on any state that is not a terminal state
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not self._is_terminal(self._agent_location):
                break
        
        return self._get_obs(), self._get_info()

    def _is_terminal(self, location):
        return any(np.array_equal(location, t) for t in self._terminal_states)

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # get direction
        direction = self._action_to_direction[action]

        # update position
        self._agent_location = np.clip(
            self._agent_location + direction,0,self.size-1
        )

        # check if agent reached target
        terminated = self._is_terminal(self._agent_location)

        # we don't use truncation?
        truncated = False

        # reward is -1 on all transitions, incl the transition into the terminal
        reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
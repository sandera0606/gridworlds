from typing import Optional
import numpy as np
import gymnasium as gym

# 10x7 gridworld based on example 6.5 of http://incompleteideas.net/book/RLbook2020.pdf
class WindyGridWorld(gym.Env):
    def __init__(self, height: int=7, width: int = 10):
        # size of the square grid (10x7 by default)
        self.height = height
        self.width = width

        # starting position
        self._agent_location = np.array([0, 3])

        # set terminal state
        self._terminal_state = np.array([7, 3])

        # define what the agent can observe
        # lower bound, upper bound, shape (n dimensions)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.width - 1, self.height - 1]),
            shape=(2,),
            dtype=int
        )

        # define what actions are available (4 directions: up, down, left, right)
        self.action_space = gym.spaces.Discrete(4)

        # wind (rows)
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

        # map action numbers to actual movements on the grid
        # [row, col]
        self._action_to_direction = {
            0: np.array([1, 0]), # move right
            1: np.array([0, 1]), # move up
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # down
        }

    def _apply_wind(self):
        col = self._agent_location[0]
        wind_strength = self.wind[col]
        self._agent_location = np.clip(
            self._agent_location + np.array([0, wind_strength]),
            np.array([0, 0]),
            np.array([self.width - 1, self.height - 1]),
        )

    def _get_obs(self) -> np.ndarray:
        """
            convert states to available observations

            Returns:
                array: Observation with agent position
        """
        return self._agent_location.copy()
    
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

        # starting position
        self._agent_location = np.array([0, 3])
        
        return self._get_obs(), self._get_info()

    def _is_terminal(self, location):
        return np.array_equal(location, self._terminal_state)

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
        # np.clip keeps it inside the grid
        self._agent_location = np.clip(
            self._agent_location + direction, np.array([0, 0]), np.array([self.width -1, self.height -1])
        )

        # apply wind
        self._apply_wind()

        # check if agent reached target
        terminated = self._is_terminal(self._agent_location)

        # we don't use truncation?
        truncated = False

        # reward is -1 on all transitions, incl the transition into the terminal
        reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
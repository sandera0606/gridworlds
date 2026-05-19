from typing import Optional
import numpy as np
import gymnasium as gym

# cliff walking example based on section 6.6 of http://incompleteideas.net/book/RLbook2020.pdf
class Cliff(gym.Env):
    def __init__(self, width = 12, height = 4):
        # size of the square grid (12x4 by default)
        self.width = width
        self.height = height

        # start position
        self._agent_location = np.array([0, 0])

        self._start = np.array([0, 0])

        # set terminal position
        self._terminal_state = np.array([self.width-1, 0])

        # define what the agent can observe - self._agent_location
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.width -1, self.height-1]),
            shape=(2,),
            dtype=int
        )

        # define what actions are available (4 directions: up, down, left, right)
        self.action_space = gym.spaces.Discrete(4)

        # map action numbers to actual movements on the grid
        # [col, row]
        self._action_to_direction = {
            0: np.array([1, 0]), # move right
            1: np.array([0, 1]), # move up
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # down
        }

        self._cliff = range(1, self.width-1)
    
    def _check_cliff(self):
        if self._agent_location[1] == 0 and self._agent_location[0] in self._cliff:
            self._agent_location = self._start.copy()
            return True
        return False

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

        self._agent_location = self._start.copy()
        
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
        self._agent_location = np.clip(
            self._agent_location + direction,np.array([0, 0]), np.array([self.width - 1, self.height-1])
        )

        # reward is -100 on cliff, -1 on all other transitions, incl the transition into the terminal
        reward = -100 if self._check_cliff() else -1
    
         # check if agent reached target
        terminated = self._is_terminal(self._agent_location)

        # we don't use truncation?
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
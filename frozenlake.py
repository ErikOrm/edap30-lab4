import copy
from typing import Any, override, cast

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt


class FrozenLakeEnv(gym.Env):

    @override
    def __init__(self, map_size: int = 8, seed: int = 0, is_slippery=False) -> None:
        self.map_size = map_size
        self._env = gym.make(
            'FrozenLake-v1',
            desc=generate_random_map(size=8, seed=seed),
            render_mode="rgb_array",
            is_slippery=is_slippery
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.is_slippery = is_slippery

    def step(self, action: int) -> tuple[int, float, bool]:
        obs, reward, terminated, truncated, _info = self._env.step(action)
        if terminated and not reward == 1:
            reward = -1
        return cast(int, obs), cast(float, reward), terminated

    def reset(
            self,
            *, # This means that seed and options are only allowed to be set by keyword
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ):
        return self._env.reset(seed=seed, options=options)

    def render(self):
        plt.imshow(self._env.render())

    def render_episode(
            self,
            states: list[tuple[int, int]]
    ):
        copy_env = copy.deepcopy(self._env)
        copy_env.reset()
        copy_env.render()
        for state in states:
            copy_env._state = state
            copy_env.render()

    def plot_value_function(self, Q: npt.NDArray):
        V = np.max(Q, axis=1).reshape(self.map_size, self.map_size)

        layout = [[cell.decode("utf-8") for cell in row] for row in self._env.env.env.env.desc]
        hole_positions = [
            (i, j)
            for i, row in enumerate(layout)
            for j, val in enumerate(row)
            if val == 'H'
        ]
        for i, j in hole_positions:
            V[i, j] = -1
        V[-1, -1] = 1
        plt.imshow(V, cmap="Greys")
        plt.colorbar()

    def get_optimal_policy(self):

        """
        WIP. The idea was to calculate the optimal policy given that we cheat and know the underlying MDP using value iteration. Not finished yet.
        """

        # First we get the transition matrix
        action_map = {
            0: (0, -1),  # Left
            1: (1, 0),  # Down
            2: (0, 1),  # Right
            3: (-1, 0)  # Up
        }
        T = np.zeros((self.map_size ** 2, 4, self.map_size ** 2))

        def _update(_state, _action, _value):
            row, col = divmod(_state, self.map_size)
            dx, dy = action_map[_action]
            new_row = np.clip(row + dx, 0, self.map_size - 1)
            new_col = np.clip(col + dy, 0, self.map_size - 1)
            _new_state = new_row * self.map_size + new_col

            # Deterministic transition
            return _new_state

        if not self.is_slippery:
            for state in range(self.map_size ** 2):
                for action in range(4):
                    new_state = _update(state, action, state)
                    T[state, action, new_state] = 1
        else:
            for state in range(self.map_size ** 2):
                for action in range(4):
                    for diff in [-1, 0, 1]:
                        mod_action = action + diff % 4
                        new_state = _update(state, mod_action, state)
                        T[state, action, new_state] = 1/3

        print("Shape of transition matrix:", T.shape)



from collections import defaultdict
import numpy as np


class GridWorld:
    def __init__(self):
        self.action_space = {
            "UP": np.array([-1, 0]),
            "RIGHT": np.array([0, 1]),
            "DOWN": np.array([1, 0]),
            "LEFT": np.array([0, -1]),
        }
        self.reward_map = np.array(
            [[0, 0, None, 0, 10],
             [0, None, -5, 0, 0],
             [0, 0, 0, None, 0],
             [-5, 0, None, 0, 0],
             [0, 0, 0, 0, -5]]
        )
        self.start_state = np.array([0, 0])
        self.goal_state = np.array([0, 4])
        self.wall_states = np.array([[0, 2], [1, 1], [2, 3], [3, 2]])
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        move = self.action_space[action]
        next_state = state + move
        if (next_state[0] < 0 or next_state[0] > env.height
            or next_state[1] < 0 or next_state[1] > env.width
                or next_state is None):
            next_state = state

        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if (state == env.goal_state):
            V[state] = 0
            continue

        action_probs = pi[state]

env = GridWorld()
V = defaultdict(lambda: 0)

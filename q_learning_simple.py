from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.actionsize = 4
        self.Q = defaultdict(lambda: 0)
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actionsize)
        else:
            qs = [self.Q[state, a] for a in range(self.actionsize)]
            return np.argmax(qs)
    def update(self, state, action, reward, next_state, done):
        if done:
            nextqmax = 0
        else:
            nextqs = [self.Q[next_state, a] for a in range(self.actionsize)]
            nextqmax = max(nextqs)
        target = self.gamma * nextqmax + reward
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

env = GridWorld()
agent = QLearningAgent()
episodes = 10000
for episode in range(episodes):
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state
env.render_q(agent.Q)
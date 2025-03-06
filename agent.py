"""
Q-learning agent for the gridworld environment.
Written by: Rowan Rosenberg March 2025
"""

import random
import pickle

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001):
        """
        actions: list of possible actions (e.g., ['up', 'down', 'left', 'right'])
        alpha: learning rate
        gamma: discount factor
        epsilon: initial exploration rate
        epsilon_decay: multiplicative decay factor applied to epsilon after each episode
        epsilon_min: minimum exploration rate
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = {}  # Q-table: dictionary mapping state -> {action: value}

    def _ensure_state(self, state):
        """Initialize Q-values for unseen states."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}

    def choose_action(self, state):
        """ Choose an action using an epsilon-greedy strategy."""
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Return the action with the highest Q-value; if tie, select randomly.
            max_value = max(self.Q[state].values())
            best_actions = [action for action, value in self.Q[state].items() if value == max_value]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """ Update the Q-table using the Q-learning update rule."""
        self._ensure_state(state)
        self._ensure_state(next_state)
        current_Q = self.Q[state][action]
        # If the next state is terminal, there is no next Q-value.
        max_next_Q = 0.0 if done else max(self.Q[next_state].values())
        # Q-learning update
        self.Q[state][action] = current_Q + self.alpha * (reward + self.gamma * max_next_Q - current_Q)

    def update_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """Save the Q-table to a file using pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self, filename):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)
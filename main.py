import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_action_values = np.random.randn(num_arms)  # True action values
        self.estimated_action_values = np.zeros(num_arms)  # Estimated action values
        self.action_counts = np.zeros(num_arms)  # Action counts
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            # Choose a random action (exploration)
            return np.random.choice(self.num_arms)
        else:
            # Choose the action with the highest estimated action value (exploitation)
            return np.argmax(self.estimated_action_values)

    def update_action_value(self, action, reward):
        self.action_counts[action] += 1
        self.estimated_action_values[action] += (reward - self.estimated_action_values[action]) / self.action_counts[action]

def run_bandit(num_arms, num_steps):
    bandit = MultiArmedBandit(num_arms)
    rewards = []

    for step in range(num_steps):
        action = bandit.choose_action()
        reward = np.random.normal(loc=bandit.true_action_values[action], scale=1.0)
        bandit.update_action_value(action, reward)
        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    num_arms = 5  # Number of arms (actions)
    num_steps = 500  # Number of steps (time steps)

    rewards = run_bandit(num_arms, num_steps)

    # Plot the rewards over time
    plt.plot(rewards)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.show()

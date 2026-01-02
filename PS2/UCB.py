import numpy as np
import random
import math

class UCB:
    def __init__(self, n_arms, c=2.0):
        """
        Args:
            n_arms: Number of arms/bandits
            c: Exploration parameter (default=2.0)
        """
        self.n_arms = n_arms
        self.c = c
        # Q-values (average rewards)
        self.q_values = np.zeros(n_arms)
        # Count of selections for each arm
        self.arm_counts = np.zeros(n_arms, dtype=int)
        # Total steps taken
        self.total_steps = 0
        # Track total reward
        self.total_reward = 0

    def select_arm(self):
        """
        Select arm using UCB formula:
        argmax [ Q(a) + c * sqrt(ln(t) / N(a)) ]
        """
        # To make sure all arms are tried atleast once before exploration
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                return arm
        
        # Calculate UCB values for all arms
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
            exploration_term = self.c * math.sqrt(math.log(self.total_steps) / self.arm_counts[arm])
            ucb_values[arm] = self.q_values[arm] + exploration_term
        
        # Choose arm with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        """
        Update the Q-value estimate for the chosen arm
        """
        # Update counts
        self.arm_counts[chosen_arm] += 1
        self.total_steps += 1
        
        # Update Q-value using incremental average
        n = self.arm_counts[chosen_arm]
        self.q_values[chosen_arm] = self.q_values[chosen_arm] + (1/n) * (reward - self.q_values[chosen_arm])
        
        # Track total reward
        self.total_reward += reward
    
    def get_best_arm(self):
        return np.argmax(self.q_values)
    
def test_ucb():
    
    bandit = UCB(n_arms=3, c=2.0)
    
    for i in range(15):
        arm = bandit.select_arm()
        reward = random.choice([0, 1])  # Random 0 or 1
        bandit.update(arm, reward)
        
        print(f"\nStep {i+1}:")
        print(f"  Selected arm: {arm}")
        print(f"  Reward: {reward}")
        print(f"  Q-values (avg rewards): {bandit.q_values.round(3)}")
        print(f"  Arm counts: {bandit.arm_counts}")
        
    print("\n" + "="*40)
    print("FINAL RESULTS:")
    print(f"Total steps: {bandit.total_steps}")
    print(f"Total reward: {bandit.total_reward}")
    print(f"Average reward: {bandit.total_reward/bandit.total_steps:.3f}")
    print(f"\nFinal Q-values: {bandit.q_values}")
    print(f"Arm counts: {bandit.arm_counts}")
    print(f"Best arm (simple regret): {bandit.get_best_arm()}")

if __name__ == "__main__":
    test_ucb()
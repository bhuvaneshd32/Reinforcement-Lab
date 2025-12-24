import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize Epsilon-Greedy algorithm
        
        Args:
            n_arms: Number of arms/bandits
            epsilon: Exploration rate (0 < epsilon < 1)
        """

        self.n_arms = n_arms
        self.epsilon = epsilon

        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zero(n_arms,dtype=int)

        # Optional: Track total reward and steps
        self.total_reward = 0
        self.steps = 0
    
    def select_arm(self):
        # epsilon greedy stratergy

        if random.random < self.epsilon :
            #Explore : Choose random arm 
            return random.radint(0,self.arms-1)
        else :
            # EXPLOIT: choose arm with highest Q-value
            # If multiple arms have same max value, choose randomly among them
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values==max_q)[0]
            
            return random.choice(best_arms)
        
    def update(self,chosen_arm,reward):
        """
        Update the Q-value estimate for the chosen arm
        
        Args:
            chosen_arm: Index of the arm that was played
            reward: Observed reward
        """
        #increment the chosen arm
        self.arm_counts[chosen_arm] +=1

        # Q-value incremental average formula

        self.q_values[chosen_arm] += ((1/self.arm_counts[chosen_arm]) * (reward- self.q_values[chosen_arm]))
        
        # Track statistics
        self.total_reward += reward
        self.steps += 1   
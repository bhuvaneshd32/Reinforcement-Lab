import numpy as np
import random

class IncrementalUniformEnhanced:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0
        self.rounds_completed = 0
        
        # For tracking selection order
        self.selection_order = []
        self.next_arm_index = 0
    
    def select_arm(self):
        """Pull arms in strict round-robin order"""
        # Always pull the next arm in sequence
        arm_to_pull = self.next_arm_index
        # Update for next selection
        self.next_arm_index = (self.next_arm_index + 1) % self.n_arms
        # Track selection
        self.selection_order.append(arm_to_pull)
        
        return arm_to_pull
    
    def update(self, chosen_arm, reward):
        """Update statistics for the chosen arm"""
        self.arm_counts[chosen_arm] += 1
        self.total_pulls += 1
        
        # Update average reward (running average)
        n = self.arm_counts[chosen_arm]
        self.q_values[chosen_arm] = self.q_values[chosen_arm] + (1/n) * (reward - self.q_values[chosen_arm])
        
        # Track when we complete a full round
        if self.total_pulls % self.n_arms == 0:
            self.rounds_completed += 1
    
    def get_best_arm(self):
        """Return arm with highest average reward"""
        return np.argmax(self.q_values)
    
    def get_stats(self):
        """Return algorithm statistics"""
        return {
            'q_values': self.q_values.copy(),
            'arm_counts': self.arm_counts.copy(),
            'total_pulls': self.total_pulls,
            'rounds_completed': self.rounds_completed,
            'selection_order': self.selection_order.copy()
        }

def demo_incremental_uniform():
    """Demonstrate how Incremental Uniform works"""
    print("=== Incremental Uniform Demonstration ===")
    print("Key property: Arms are pulled in strict round-robin order.")
    print("Each arm gets pulled exactly once per round.\n")
    
    bandit = IncrementalUniformEnhanced(n_arms=3)
    
    # True reward probabilities (unknown to algorithm)
    true_probs = [0.9, 0.3, 0.6]
    
    print(f"True probabilities: Arm 0: {true_probs[0]}, Arm 1: {true_probs[1]}, Arm 2: {true_probs[2]}")
    print(f"Expected: Arm 0 should emerge as best after several rounds.\n")
    
    # Run 12 steps (4 complete rounds)
    for step in range(12):
        arm = bandit.select_arm()
        
        # Simulate reward based on true probability
        reward = 1 if random.random() < true_probs[arm] else 0
        
        bandit.update(arm, reward)
        
        print(f"Step {step+1:2d}: Pulled arm {arm}, Reward {reward}")
        
        # Show state after each complete round
        if (step + 1) % 3 == 0:  # After each round of 3 arms
            stats = bandit.get_stats()
            print(f"  [End of Round {stats['rounds_completed']}]")
            print(f"  Arm counts: {stats['arm_counts']}")
            print(f"  Q-values: {[f'{q:.3f}' for q in stats['q_values']]}")
            print(f"  Current best arm: {bandit.get_best_arm()}")
            print()
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    stats = bandit.get_stats()
    
    print(f"\nTotal pulls: {stats['total_pulls']}")
    print(f"Rounds completed: {stats['rounds_completed']}")
    print(f"\nArm statistics:")
    for i in range(bandit.n_arms):
        print(f"  Arm {i}: Pulled {stats['arm_counts'][i]} times, "
              f"Avg reward = {stats['q_values'][i]:.3f}")
    
    print(f"\nBest arm identified: {bandit.get_best_arm()}")
    print(f"Difference between max and min pulls: "
          f"{max(stats['arm_counts']) - min(stats['arm_counts'])}")
    
    print(f"\nSelection order (first 12): {stats['selection_order'][:12]}")
    print("Pattern: 0, 1, 2, 0, 1, 2, 0, 1, 2, ... (perfect round-robin)")

if __name__ == "__main__":
    demo_incremental_uniform()
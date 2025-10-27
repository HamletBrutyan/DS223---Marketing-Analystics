"""
Multi-Armed Bandit Algorithms for A/B Testing
=============================================

This module implements two bandit algorithms for solving the explore-exploit dilemma
in the context of A/B testing for advertisement optimization:

1. **Epsilon-Greedy**: A strategy that balances exploration and exploitation
   - Initial epsilon: 0.1
   - Epsilon decays by 1/t during the experiment
   
2. **Thompson Sampling**: A Bayesian approach using Gaussian posterior distributions
   - Uses Gaussian distributions with known precision (tau = 100.0)
   - Updates posterior distributions using Gaussian-Gaussian conjugacy

Experiment Configuration:
------------------------
- Number of Bandits: 4 
- Bandit Rewards: [1, 2, 3, 4]
- Number of Trials: 20,000
- Reward Distribution: Gaussian with noise (standard deviation = 0.1 for epsilon-greedy)

Key Features:
------------
- Proper implementation of abstract Bandit methods (pull, update, experiment, report)
- Comprehensive data logging to CSV files with format: Bandit, Reward, Algorithm
- Visualization of learning process and cumulative metrics
- Automatic calculation of cumulative rewards and cumulative regrets

Visualization:
-------------
- plot1(): Learning process visualization (average reward over time)
- plot2(): Cumulative rewards and regrets comparison
- Output files: learning_process.png, cumulative_metrics.png
- CSV files: epsilon_greedy_results.csv, thompson_sampling_results.csv

Usage:
------
    from Bandit import comparison
    comparison()

Author: Hamlet Brutyan
Course: Marketing Analytics
Date: 26/10/2025

Implementation Improvement Strategies:
------------------------------------
1. Adaptive Epsilon - Sophisticated decay schedules (exponential, learning-based)
2. Bayesian Optimization - Auto-tune hyperparameters (epsilon, precision)
3. Restless Bandits - Handle arms with changing reward distributions over time
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """Visualization class for bandit algorithm results
    
    This class provides visualization methods for analyzing bandit performance:
    - plot1(): Shows learning process (average reward over time)
    - plot2(): Compares cumulative rewards and regrets across algorithms
    """
    
    @staticmethod
    def plot1(epsilon_greedy, thompson_sampling, num_trials):
        """Visualize the learning process for each algorithm
        
        Shows how average reward changes over time for both algorithms,
        displayed on both linear and logarithmic scales.
        
        Parameters
        ----------
        epsilon_greedy : EpsilonGreedy
            EpsilonGreedy bandit object with experiment results
        thompson_sampling : ThompsonSampling
            ThompsonSampling bandit object with experiment results
        num_trials : int
            Number of trials in the experiment
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calculate running averages (learning process)
        eg_running_avg = np.array(epsilon_greedy.cumulative_rewards) / np.arange(1, num_trials + 1)
        ts_running_avg = np.array(thompson_sampling.cumulative_rewards) / np.arange(1, num_trials + 1)
        
        # Optimal average reward line
        optimal_avg = max(epsilon_greedy.p)
        
        # Linear scale plot
        axes[0].plot(eg_running_avg, label='Epsilon-Greedy', linewidth=1.5, alpha=0.8)
        axes[0].plot(ts_running_avg, label='Thompson Sampling', linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=optimal_avg, color='r', linestyle='--', label='Optimal', linewidth=2)
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('Average Reward', fontsize=12)
        axes[0].set_title('Learning Process (Linear Scale)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Log scale plot
        axes[1].plot(eg_running_avg, label='Epsilon-Greedy', linewidth=1.5, alpha=0.8)
        axes[1].plot(ts_running_avg, label='Thompson Sampling', linewidth=1.5, alpha=0.8)
        axes[1].axhline(y=optimal_avg, color='r', linestyle='--', label='Optimal', linewidth=2)
        axes[1].set_xlabel('Trial', fontsize=12)
        axes[1].set_ylabel('Average Reward', fontsize=12)
        axes[1].set_title('Learning Process (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_process.png', dpi=300, bbox_inches='tight')
        logger.info("Learning process plots saved as 'learning_process.png'")
        plt.show()

    @staticmethod
    def plot2(epsilon_greedy, thompson_sampling):
        """Compare cumulative rewards and regrets across algorithms
        
        Shows cumulative metrics for both Epsilon-Greedy and Thompson Sampling
        to visualize algorithm performance comparison.
        
        Parameters
        ----------
        epsilon_greedy : EpsilonGreedy
            EpsilonGreedy bandit object with experiment results
        thompson_sampling : ThompsonSampling
            ThompsonSampling bandit object with experiment results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cumulative rewards comparison
        axes[0].plot(epsilon_greedy.cumulative_rewards, label='Epsilon-Greedy', linewidth=2, alpha=0.9)
        axes[0].plot(thompson_sampling.cumulative_rewards, label='Thompson Sampling', linewidth=2, alpha=0.9)
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('Cumulative Reward', fontsize=12)
        axes[0].set_title('Cumulative Rewards Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative regrets comparison
        axes[1].plot(epsilon_greedy.cumulative_regrets, label='Epsilon-Greedy', linewidth=2, alpha=0.9)
        axes[1].plot(thompson_sampling.cumulative_regrets, label='Thompson Sampling', linewidth=2, alpha=0.9)
        axes[1].set_xlabel('Trial', fontsize=12)
        axes[1].set_ylabel('Cumulative Regret', fontsize=12)
        axes[1].set_title('Cumulative Regrets Comparison', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cumulative_metrics.png', dpi=300, bbox_inches='tight')
        logger.info("Cumulative metrics plots saved as 'cumulative_metrics.png'")
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy Multi-Armed Bandit Algorithm
    
    This algorithm balances exploration and exploitation using an epsilon parameter.
    With probability epsilon, it explores by choosing a random arm. Otherwise, 
    it exploits by choosing the arm with the highest estimated reward.
    
    Attributes:
        p (list): True success probabilities for each arm
        epsilon (float): Exploration parameter (0 to 1)
        num_trials (int): Number of trials to run
        p_estimate (list): Estimated probabilities for each arm
        N (list): Number of times each arm was pulled
        rewards (list): Reward values for each trial
        regrets (list): Regret values for each trial
        cumulative_rewards (list): Cumulative rewards
        cumulative_regrets (list): Cumulative regrets
        optimal (int): Index of the optimal arm
    """
    
    def __init__(self, p, epsilon=0.1):
        """Initialize EpsilonGreedy bandit
        
        Parameters
        ----------
        p : list of float
            True rewards (expected values) for each bandit arm
        epsilon : float, optional
            Initial exploration parameter (default is 0.1)
            Note: Epsilon decays by 1/t during experiment
        """
        self.p = p
        self.epsilon_initial = epsilon  # Store initial value
        self.num_trials = 20000
        self.p_estimate = [0.0] * len(p)
        self.N = [0] * len(p)
        self.rewards = []
        self.regrets = []
        self.cumulative_rewards = []
        self.cumulative_regrets = []
        self.optimal = np.argmax(p)
        self.chosen_arms = []  # Track which arm was chosen each trial
        
    def __repr__(self):
        return f"EpsilonGreedy with initial epsilon={self.epsilon_initial}, optimal arm={self.optimal}"
    
    def pull(self):
        """Pull the currently active arm and return the reward
        
        This method pulls the arm specified by self.current_arm_idx and returns
        the reward value (with added Gaussian noise for realism).
        
        Returns
        -------
        float
            Reward value with added noise
            
        Note
        ----
        You must set self.current_arm_idx before calling this method
        """
        # Add Gaussian noise to the reward (standard deviation = 0.1)
        noise = np.random.normal(0, 0.1)
        return self.p[self.current_arm_idx] + noise
    
    def update(self):
        """Update the estimate for the arm that was just pulled
        
        This method updates the estimate for the arm specified by self.current_arm_idx
        based on the reward stored in self.current_reward.
        
        Note
        ----
        You must set self.current_arm_idx and self.current_reward before calling this method
        """
        arm_idx = self.current_arm_idx
        reward = self.current_reward
        self.N[arm_idx] += 1
        self.p_estimate[arm_idx] = ((self.N[arm_idx] - 1) * self.p_estimate[arm_idx] + reward) / self.N[arm_idx]
    
    def experiment(self):
        """Run the Epsilon-Greedy experiment
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        cumulative_reward = 0
        cumulative_regret = 0
        
        for i in range(self.num_trials):
            # Epsilon decay: epsilon = epsilon_initial / (t+1)
            t = i + 1
            epsilon_t = self.epsilon_initial / t
            
            # Choose action: explore or exploit
            if np.random.random() < epsilon_t:
                # Explore: choose random arm
                chosen_arm = np.random.randint(len(self.p))
            else:
                # Exploit: choose arm with highest estimate
                chosen_arm = np.argmax(self.p_estimate)
            
            self.chosen_arms.append(chosen_arm)
            
            # Pull the chosen arm
            self.current_arm_idx = chosen_arm
            reward = self.pull()
            self.rewards.append(reward)
            cumulative_reward += reward
            self.cumulative_rewards.append(cumulative_reward)
            
            # Update estimates
            self.current_reward = reward
            self.update()
            
            # Calculate regret (difference between optimal and chosen arm reward)
            optimal_reward = self.p[self.optimal]
            chosen_reward = self.p[chosen_arm]
            regret = optimal_reward - chosen_reward
            self.regrets.append(regret)
            cumulative_regret += regret
            self.cumulative_regrets.append(cumulative_regret)
        
        return self
    
    def report(self):
        """Report results to CSV and print statistics
        
        This method saves the experiment results to a CSV file and logs
        key statistics including average reward and regret.
        """
        # Create DataFrame with required format: Bandit, Reward, Algorithm
        df = pd.DataFrame({
            'Bandit': self.chosen_arms,
            'Reward': self.rewards,
            'Algorithm': ['Epsilon-Greedy'] * self.num_trials
        })
        df.to_csv('epsilon_greedy_results.csv', index=False)
        
        avg_reward = np.mean(self.rewards)
        avg_regret = np.mean(self.regrets)
        total_reward = sum(self.rewards)
        optimal_total = self.num_trials * self.p[self.optimal]
        
        logger.info(f"=== EpsilonGreedy Results ===")
        logger.info(f"Average Reward: {avg_reward:.6f}")
        logger.info(f"Average Regret: {avg_regret:.6f}")
        logger.info(f"Total Reward: {total_reward:.0f}")
        logger.info(f"Optimal Total: {optimal_total:.0f}")
        logger.info(f"Efficiency: {total_reward/optimal_total:.4f}")
        logger.info(f"Number of times each arm was pulled: {self.N}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling Multi-Armed Bandit Algorithm (Gaussian)
    
    This algorithm uses Bayesian posterior distributions (Gaussian) with known precision.
    The precision parameter represents the inverse variance of the noise.
    It samples from the posterior distribution of each arm and selects the arm
    with the highest sample value.
    
    Attributes:
        p (list): True expected rewards for each arm
        num_trials (int): Number of trials to run
        tau (float): Known precision (inverse variance)
        mu (list): Posterior means for each arm
        N (list): Number of times each arm was pulled
        rewards (list): Reward values for each trial
        regrets (list): Regret values for each trial
        cumulative_rewards (list): Cumulative rewards
        cumulative_regrets (list): Cumulative regrets
        optimal (int): Index of the optimal arm
    """
    
    def __init__(self, p, precision=100.0):
        """Initialize ThompsonSampling bandit with Gaussian distributions
        
        Parameters
        ----------
        p : list of float
            True expected rewards for each bandit arm
        precision : float, optional
            Known precision (inverse variance) of the reward noise (default is 100.0)
            Note: Higher precision means lower variance (less noise)
        """
        self.p = p
        self.precision = precision  # tau = 1/sigma^2
        self.num_trials = 20000
        # Gaussian posterior parameters: mu ~ N(m, lambda)
        self.mu = [0.0] * len(p)  # Posterior means
        self.lambda_ = [1.0] * len(p)  # Precision (inverse variance) for each arm
        self.N = [0] * len(p)
        self.rewards = []
        self.regrets = []
        self.cumulative_rewards = []
        self.cumulative_regrets = []
        self.optimal = np.argmax(p)
        self.chosen_arms = []  # Track which arm was chosen each trial
        
    def __repr__(self):
        return f"ThompsonSampling bandit (precision={self.precision}), optimal arm={self.optimal}"
    
    def pull(self):
        """Pull the currently active arm and return the reward
        
        This method pulls the arm specified by self.current_arm_idx and returns
        the reward value with Gaussian noise.
        
        Returns
        -------
        float
            Reward value with Gaussian noise
            
        Note
        ----
        You must set self.current_arm_idx before calling this method
        """
        # Add Gaussian noise with known precision (tau)
        sigma = 1.0 / np.sqrt(self.precision)
        noise = np.random.normal(0, sigma)
        return self.p[self.current_arm_idx] + noise
    
    def sample(self, arm_idx):
        """Sample from posterior Gaussian distribution for an arm
        
        Parameters
        ----------
        arm_idx : int
            Index of the arm to sample from
            
        Returns
        -------
        float
            Sample value from N(mu[arm_idx], 1/lambda[arm_idx])
        """
        # Sample from Gaussian posterior
        sigma = 1.0 / np.sqrt(self.lambda_[arm_idx])
        return np.random.normal(self.mu[arm_idx], sigma)
    
    def update(self):
        """Update the Gaussian posterior distribution for the arm that was just pulled
        
        For Gaussian-Gaussian conjugacy with known precision tau:
        - mu_new = (tau * x + lambda_old * mu_old) / (tau + lambda_old)
        - lambda_new = lambda_old + tau
        
        Note
        ----
        You must set self.current_arm_idx and self.current_reward before calling this method
        """
        arm_idx = self.current_arm_idx
        reward = self.current_reward
        
        # Update Gaussian posterior
        self.mu[arm_idx] = (self.precision * reward + self.lambda_[arm_idx] * self.mu[arm_idx]) / (self.precision + self.lambda_[arm_idx])
        self.lambda_[arm_idx] += self.precision
        self.N[arm_idx] += 1
    
    def experiment(self):
        """Run the Thompson Sampling experiment
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        cumulative_reward = 0
        cumulative_regret = 0
        
        for i in range(self.num_trials):
            # Sample from each arm's posterior distribution
            samples = [self.sample(j) for j in range(len(self.p))]
            
            # Choose arm with highest sample (Thompson Sampling decision)
            chosen_arm = np.argmax(samples)
            self.chosen_arms.append(chosen_arm)
            
            # Pull the chosen arm
            self.current_arm_idx = chosen_arm
            reward = self.pull()
            self.rewards.append(reward)
            cumulative_reward += reward
            self.cumulative_rewards.append(cumulative_reward)
            
            # Update posterior distribution
            self.current_reward = reward
            self.update()
            
            # Calculate regret (difference between optimal and chosen arm expected reward)
            optimal_reward = self.p[self.optimal]
            chosen_reward = self.p[chosen_arm]
            regret = optimal_reward - chosen_reward
            self.regrets.append(regret)
            cumulative_regret += regret
            self.cumulative_regrets.append(cumulative_regret)
        
        return self
    
    def report(self):
        """Report results to CSV and print statistics
        
        This method saves the experiment results to a CSV file and logs
        key statistics including average reward and regret.
        """
        # Create DataFrame with required format: Bandit, Reward, Algorithm
        df = pd.DataFrame({
            'Bandit': self.chosen_arms,
            'Reward': self.rewards,
            'Algorithm': ['Thompson Sampling'] * self.num_trials
        })
        df.to_csv('thompson_sampling_results.csv', index=False)
        
        avg_reward = np.mean(self.rewards)
        avg_regret = np.mean(self.regrets)
        total_reward = sum(self.rewards)
        optimal_total = self.num_trials * self.p[self.optimal]
        
        logger.info(f"=== ThompsonSampling Results ===")
        logger.info(f"Average Reward: {avg_reward:.6f}")
        logger.info(f"Average Regret: {avg_regret:.6f}")
        logger.info(f"Total Reward: {total_reward:.0f}")
        logger.info(f"Optimal Total: {optimal_total:.0f}")
        logger.info(f"Efficiency: {total_reward/optimal_total:.4f}")
        logger.info(f"Number of times each arm was pulled: {self.N}")




def comparison():
    """Compare EpsilonGreedy and ThompsonSampling algorithms
    
    This function runs experiments with both algorithms and generates all required
    visualizations and reports as specified in the homework requirements.
    
    Parameters:
    - Bandit rewards: [1, 2, 3, 4]
    - Number of trials: 20,000
    - Epsilon-Greedy: epsilon_initial = 0.1, decays by 1/t
    - Thompson Sampling: precision = 100.0 (known precision, Gaussian)
    
    Outputs:
    - CSV files with format: Bandit, Reward, Algorithm
    - Learning process plots (plot1)
    - Cumulative metrics plots (plot2)
    - Console output with cumulative reward and cumulative regret
    """
    # Experiment parameters
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    
    # Algorithm parameters
    epsilon_initial = 0.1  # Initial epsilon, decays by 1/t during experiment
    precision = 100.0  # Known precision (tau) for Thompson Sampling (inverse variance)
    
    logger.info("="*60)
    logger.info("A/B TESTING EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Bandit Rewards: {bandit_rewards}")
    logger.info(f"Number of Trials: {num_trials}")
    logger.info(f"Epsilon-Greedy: epsilon_initial = {epsilon_initial} (decays by 1/t)")
    logger.info(f"Thompson Sampling: precision = {precision}")
    logger.info("="*60)
    
    # Initialize bandits
    eg = EpsilonGreedy(bandit_rewards, epsilon=epsilon_initial)
    ts = ThompsonSampling(bandit_rewards, precision=precision)
    
    # Run experiments
    logger.info("\nRunning EpsilonGreedy experiment...")
    eg.experiment()
    
    logger.info("\nRunning ThompsonSampling experiment...")
    ts.experiment()
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("EPSILON-GREEDY RESULTS")
    logger.info("="*60)
    eg.report()
    
    logger.info("\n" + "="*60)
    logger.info("THOMPSON SAMPLING RESULTS")
    logger.info("="*60)
    ts.report()
    
    # Print cumulative reward and cumulative regret as required
    logger.info("\n" + "="*60)
    logger.info("FINAL CUMULATIVE METRICS")
    logger.info("="*60)
    logger.info(f"Epsilon-Greedy - Cumulative Reward: {eg.cumulative_rewards[-1]:.2f}")
    logger.info(f"Epsilon-Greedy - Cumulative Regret: {eg.cumulative_regrets[-1]:.2f}")
    logger.info(f"Thompson Sampling - Cumulative Reward: {ts.cumulative_rewards[-1]:.2f}")
    logger.info(f"Thompson Sampling - Cumulative Regret: {ts.cumulative_regrets[-1]:.2f}")
    logger.info("="*60)
    
    # Generate visualizations
    viz = Visualization()
    logger.info("\nGenerating learning process plots...")
    viz.plot1(eg, ts, num_trials)
    
    logger.info("\nGenerating cumulative metrics plots...")
    viz.plot2(eg, ts)


if __name__=='__main__':
   
    logger.debug("Starting bandit algorithms...")
    logger.info("Running comparison of EpsilonGreedy and ThompsonSampling")
    
    comparison()
    
    logger.info("Analysis complete!")

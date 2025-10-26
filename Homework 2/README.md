# Multi-Armed Bandit Algorithms for A/B Testing

## Overview

This project implements two bandit algorithms for solving the explore-exploit dilemma in the context of A/B testing for advertisement optimization:

1. **Epsilon-Greedy**: A strategy that balances exploration and exploitation with decaying epsilon
2. **Thompson Sampling**: A Bayesian approach using Gaussian posterior distributions

## Experiment Configuration

- **Number of Bandits**: 4 advertisement options
- **Bandit Rewards**: [1, 2, 3, 4]
- **Number of Trials**: 20,000
- **Epsilon-Greedy**: 
  - Initial epsilon: 0.1
  - Epsilon decays by 1/t during the experiment
- **Thompson Sampling**: 
  - Precision (τ): 100.0 (known precision parameter)
  - Uses Gaussian-Gaussian conjugacy for posterior updates

## Installation


### Setup

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The requirements include:
- `loguru==0.7.2` - Logging framework
- `numpy==1.24.3` - Numerical operations
- `matplotlib==3.7.1` - Visualization
- `scipy==1.10.1` - Statistical functions
- `pandas==2.0.2` - Data processing

## Usage

### Running the Experiment

Simply run the main script:
```bash
python3 Bandit.py
```

This will:
1. Run both Epsilon-Greedy and Thompson Sampling experiments with 20,000 trials each
2. Generate visualization plots
3. Save results to CSV files
4. Display cumulative metrics to the console

### Output Files

The script generates the following output files:

#### CSV Files:
- `epsilon_greedy_results.csv` - Results from Epsilon-Greedy algorithm
  - Format: `Bandit`, `Reward`, `Algorithm`
- `thompson_sampling_results.csv` - Results from Thompson Sampling algorithm
  - Format: `Bandit`, `Reward`, `Algorithm`

#### Visualization Files:
- `learning_process.png` - Shows the learning process for both algorithms
  - Left: Linear scale
  - Right: Logarithmic scale
- `cumulative_metrics.png` - Compares cumulative rewards and regrets
  - Left: Cumulative rewards comparison
  - Right: Cumulative regrets comparison

#### Console Output:
- Cumulative reward for each algorithm
- Cumulative regret for each algorithm
- Detailed statistics including efficiency and arm selection counts

## Algorithm Details

### Epsilon-Greedy

The Epsilon-Greedy algorithm balances exploration and exploitation:
- **Exploitation**: Chooses the arm with highest estimated reward
- **Exploration**: Randomly chooses any arm
- **Epsilon Decay**: The exploration probability ε decreases as ε(t) = 0.1/(t+1)

This decay ensures more exploration early in the experiment and more exploitation as we learn.

### Thompson Sampling

Thompson Sampling uses Bayesian inference:
- **Prior**: Gaussian distribution for each arm's expected reward
- **Posterior Update**: Updates using Gaussian-Gaussian conjugacy with known precision
- **Selection**: Samples from posterior distribution and chooses arm with highest sample

The algorithm naturally balances exploration and exploitation by sampling from uncertainty distributions.

## Results

The experiments typically show:
- **Thompson Sampling** outperforms Epsilon-Greedy in terms of cumulative reward
- **Thompson Sampling** has lower cumulative regret
- Both algorithms converge toward optimal arm selection over time
- Efficient exploration leads to higher total rewards

## File Structure

```
.
├── Bandit.py                    # Main implementation file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── epsilon_greedy_results.csv   # Generated output (after running the code)
├── thompson_sampling_results.csv # Generated output (after running the code)
├── learning_process.png         # Generated visualization (after running the code)
└── cumulative_metrics.png       # Generated visualization (after running the code)
```

## Class Structure

### Abstract Base Class: `Bandit`
Defines the interface that all bandit algorithms must implement.

### EpsilonGreedy Class
Implements the epsilon-greedy algorithm with decaying epsilon.

### ThompsonSampling Class
Implements Gaussian Thompson Sampling with known precision.

### Visualization Class
Provides methods to visualize algorithm performance:
- `plot1()`: Learning process visualization
- `plot2()`: Cumulative metrics comparison

## Key Implementation Details

1. **Abstract Methods**: Properly implements all required abstract methods from the Bandit base class
2. **State Management**: Uses instance variables to maintain state for `pull()` and `update()` methods
3. **Efficient Tracking**: Cumulative metrics are tracked incrementally (O(1) per iteration)
4. **Comprehensive Logging**: All results are logged to CSV files and displayed to console

## Experimental Design

The experiment follows the A/B testing paradigm:
- Four advertisement options are tested
- Each trial corresponds to a user interaction
- Reward represents effectiveness metric (e.g., conversion rate, revenue)
- Cumulative regret measures how much worse we performed compared to always choosing the optimal arm

## Implementation Improvement Strategies

Advanced strategies for enhancing the bandit algorithms:

1. **Adaptive Epsilon Schedule**: Use a more sophisticated decay schedule than 1/t, such as exponential decay or learning-based epsilon that adjusts based on uncertainty estimates. This would allow for more optimal exploration-exploitation balance throughout the experiment.

2. **Bayesian Optimization**: Use Gaussian processes or tree-structured Parzen estimators to automatically tune hyperparameters (epsilon, precision) based on validation performance. This eliminates manual parameter tuning and finds optimal values through automated search.

3. **Restless Multi-armed Bandits**: Extend to handle arms with changing reward distributions over time, modeling real-world dynamics like user fatigue or market conditions. This accounts for non-stationary environments where advertisement effectiveness changes over time.

## Author

**Hamlet Brutyan**  
Course: Marketing Analytics  
Date: 26/10/2025


## License

This code is developed for educational purposes as part of the Marketing Analytics course at AUA (American University of Armenia).


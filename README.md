# Reinforcement Learning on Custom Grid Environment  
**Student:** Jayed Bin Rahman – 9764659  
**Module:** 6001CEM – Individual Project  

## Project Title  
Training Reinforcement Learning Agents on a Custom Grid-Based Environment Using Gymnasium and RLlib  

## Background and Motivation  
This project investigates how reinforcement learning (RL) agents can be trained to navigate a dynamic grid-based environment containing moving obstacles. While RL has achieved success in domains such as robotics and gaming, this project specifically targets the challenge of teaching agents to learn adaptive policies under environmental randomness and uncertainty.

The motivation stems from the need to understand the practical aspects of RL agent training — particularly how modern frameworks like Gymnasium and RLlib can be used effectively to prototype, evaluate, and compare different RL algorithms. The project also draws inspiration from real-world navigation and path-planning problems in robotics and autonomous systems.

## Research Objectives  
- Build a custom GridWorld environment with moving obstacles using the Gymnasium API  
- Train and compare three RL algorithms:  
  - Tabular Q-Learning  
  - Deep Q-Networks (DQN)  
  - Proximal Policy Optimization (PPO)  
- Evaluate learning efficiency, policy stability, and overall performance  
- Analyze results using metrics such as episode return, success rate, and convergence  

## Methodology Summary  
The environment is a 10×10 grid where an agent must reach a fixed goal while avoiding a randomly moving obstacle.  

**Reward structure:**  
- +1 for reaching the goal  
- -1 for collision  
- -0.1 per step to encourage efficiency  

**Trained Agents:**  
- Tabular Q-Learning: implemented from scratch using a dictionary-based Q-table  
- DQN: trained via RLlib using a fully connected neural network  
- PPO: trained via RLlib using an actor-critic architecture  

Each agent was evaluated through 100 test episodes. A learning curve was also plotted from the training logs to analyze convergence behavior.

## Key Results  
- Environment successfully integrated using Gymnasium  
- Evaluation results after training:  
  - DQN: ~96% success rate, ~7.9 steps per success  
  - PPO: ~97% success rate, ~8.2 steps per success  
  - Q-Learning: ~98% success rate, ~8.1 steps per success  
- PPO showed the fastest convergence; Q-learning was the most conservative  
- A results plot was generated and saved as `results_plot.png`  

## Supervisor Guidance  
This research was conducted under the guidance of the project supervisor. The topic selection, methodology refinement, algorithm choice, and risk assessments were discussed and validated through a combination of face-to-face meetings and Microsoft Teams conversations. The supervisor's feedback was instrumental in shaping the research direction and ensuring academic alignment with the module brief.

## Tools and Frameworks Used  
- Python 3.11  
- Gymnasium (Farama Foundation)  
- RLlib (Ray)  
- NumPy, Matplotlib  
- Visual Studio Code  
- Miniconda3  
- GitHub  

## Repository Structure

```plaintext
RL_Project_9764659/
├── custom_env/               # Custom Gymnasium environment
├── train_qlearning.py        # Tabular Q-learning script
├── train_dqn.py              # DQN training using RLlib
├── train_ppo.py              # PPO training using RLlib
├── test_env.py               # Test script for environment sanity check
├── results_plot.png          # Learning curve plot
├── progress.csv              # Logged training data
├── 6001CEM_RL_Report.docx    # Final academic report
└── README.md                 # This file


## Conclusion  
The project demonstrates that reinforcement learning agents can effectively learn to navigate dynamic environments with stochastic obstacles. PPO and DQN displayed rapid learning and generalization ability, while tabular Q-learning converged reliably through exhaustive exploration. The use of Gymnasium and RLlib facilitated a clean and reproducible experimentation framework. The project highlights both theoretical insights and practical implementation experience in modern RL development.

<h1 align="center"> Microgrid Management with Reinforcement Learning </h1>

<p style="text-align:center;">
  <img src="images/Microgrid_DQN_PPO_logo.png" alt="ABC Logo"
       style="max-height:70vh; width:auto; display:block; margin:0 auto;">
</p>

<h3 align="center"> Comparative study of Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) versus Rule-Based Controller (RBC) in Microgrid Energy Management</h3>


## :book: Table of Contents

<details open="open">
  <summary>Table of Contents</summary>

1. [➤ About The Project](#about-the-project)
2. [➤ Folder Structure](#folder-structure)
3. [➤ Test Environment](#test-environment)
4. [➤ Implementation of DQN and PPO](#dqn-ppo)
5. [➤ Microgrid](#microgrid)
6. [➤ Pymgrid](#pymgrid)
7. [➤ Results and Discussion](#results-and-discussion)
8. [➤ Conclusion](#conclusion)
9. [➤ References](#references)

</details>

</br>

## :memo: About The Project <a id="about-the-project"></a>

This work corresponds to the **Final Degree Project (TFG)** in **Computer Engineering**. What is presented in this README is a summary of the thesis submitted for the project defense.

This project investigates the **optimization of a microgrid** using Reinforcement Learning (RL), focusing on **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**, and comparing them against a **Rule-Based Controller (RBC)** baseline. It combines theoretical study of the trial-and-error RL paradigm with practical implementation in PyTorch, simulation using **Pymgrid** (built on Gymnasium), and automated hyperparameter search with **Ray Tune / Optuna**. The workflow covers learning the paradigm, mastering the key libraries, implementing and optimizing agents (including GPU support and further optimizations such as vectorization wherever possible in both algorithms), and validating them in discrete scenarios.

Experiments show both agents learn sensible policies, yet they do **not consistently outperform** a well-designed RBC on operational cost in the tested scenarios. Recommended next steps include expanding hyperparameter searches, trying alternative algorithms or continuous-action formulations, and revising reward design to boost RL performance.

</br>

## :file_folder: Folder Structure <a id="folder-structure"></a>

    TFG_RL_Microgrid
    ├── TFG_RL_MicroGrid.ipynb
    ├── Microgrid_RL.yml
    ├── LICENSE
    ├── README.md
    │   
    ├── dqn
    |   ├── replay_memory.py
    |   └── dqn.py
    │   
    ├── ppo
    |   ├── ppo_agents.py
    |   ├── ppo_buffer.py
    |   └── ppo.py
    │
    ├── logs_tensorflow
    |   ├── dqn
    |   |   ├── escenario_0
    |   |   |   └── ...
    |   |   └── escenario_22
    |   |       └── ...
    |   └── ppo
    |       ├── escenario_0
    |       |   └── ...
    |       └── escenario_22
    |           └── ...
    |
    ├── rbc_results
    |   ├── microgrid_scenario_0.csv
    |   ├── ...
    |   └── microgrid_scenario_24.csv
    |
    ├── train_logs
    |   └── ...
    |
    └── images
        └── ...

</br>

## :computer: Test Environment <a id="test-environment"></a>

The project and algorithms were developed in a controlled environment with the following hardware and software configuration:

* Processor (CPU): Intel Core i9-12900K up to 5.2 GHz Max Turbo (16 cores, 24 threads)
* Memory (RAM): 32 GB DDR5 at 5600 MHz
* Graphics Card (GPU): NVIDIA GeForce RTX 3080 Ti with 12 GB VRAM
* Operating System: Ubuntu 24.04 (64-bit)
* Python Version: 3.12
* Key Libraries and Dependencies:

  * torch 2.7.0+cu128
  * tensorboard 2.17.1
  * pymgrid 1.2.2
  * gymnasium 1.1.1
  * ray 2.45.0
  * optuna 4.3.0
  * numpy 1.26.4
  * matplotlib 3.9.2
  * pandas 2.2.3

</br>

## :brain: Implementation of DQN and PPO <a id="dqn-ppo"></a>

The reinforcement learning algorithms **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** were implemented and adapted to the Pymgrid environments.  
To improve training efficiency and scalability, several optimizations were applied:

- **GPU Acceleration**: Both algorithms were executed on NVIDIA GPUs to significantly reduce training time compared to CPU execution.  
- **Mixed Precision Training**: By using FP16 precision (mixed precision), memory consumption was reduced and throughput was improved while maintaining stable convergence.  
- **Vectorization**: Environments were vectorized to enable parallel rollouts, allowing the agents to collect more experience per unit of time and improve sample efficiency.  

These optimizations allowed faster experimentation and better use of computational resources, making large-scale hyperparameter searches feasible within limited time and hardware budgets.

</br>

## :zap: Microgrid <a id="microgrid"></a>

A microgrid is a small-scale, localized energy system designed to provide reliable power to a specific area, such as a university campus or an industrial complex. Its primary goal is to enhance efficiency, reliability, and reduce operational costs.

A key feature of a microgrid is its ability to operate in two modes:

* **Grid-Connected Mode:** It functions while connected to the main electrical grid, allowing it to import or export power as needed.

* **Island Mode:** It can operate independently and disconnected from the main grid, which is crucial for maintaining power during outages.

A microgrid consists of distributed generation sources (like renewable energy and/or generators), energy storage systems (such as batteries), the energy demand it needs to meet, and a control system that manages energy flow. The complexity of managing these components makes it a challenging optimization problem. It also includes the main utility grid (the general electrical network), with the capability to connect to or disconnect from it when required and working in the two modes established.

</br>

## :test_tube: Pymgrid <a id="pymgrid"></a>

**Pymgrid** is the main library used in this project to simulate the behavior of microgrids.  
It is an open-source simulator developed in Python, specifically designed for research in **Artificial Intelligence** applied to microgrid management.  

One of its key features is that it inherits from **Gymnasium (OpenAI)**, meaning it provides standardized environments that are directly compatible with **Reinforcement Learning (RL)** algorithms.  

---

### Microgrid Modules

Pymgrid allows modeling a microgrid by combining different modules, each representing a component of the system:

- **RenewableModule** → Simulates uncontrollable renewable energy sources, such as photovoltaic production.  
- **BatteryModule** → Simulates batteries, a controllable module that can store (charge) or supply (discharge) energy.  
- **GensetModule** → Represents a diesel generator, a controllable energy source with associated production costs.  
- **GridModule** → Models the connection to the main power grid, allowing energy imports and exports at specific costs.  
- **LoadModule** → Simulates the energy demand that the microgrid must satisfy at each time step.  

---

### Environments for RL Agents

Pymgrid provides two types of environments to interact with RL agents, depending on the action space:

- **DiscreteMicrogridEnv**  
  - The action space is **discrete**.  
  - Actions are represented as **priority lists** that define the order in which controllable modules (battery, grid, generator) are used to meet demand.  

- **ContinuousMicrogridEnv**  
  - The action space is **continuous**.  
  - Actions represent **percentages of charging or discharging** for each controllable module.  

---

### Reward Function

Reward calculation, a crucial element in RL, is based on the **economic costs of management**.  
At each time step, the reward is the **negative sum** of all incurred costs:

- Energy import costs
- CO₂ penalty costs  
- Costs of unmet demand  
- Costs of overgeneration
- Cost of supply failure

➡️ The goal of the RL agent is to **maximize the reward**, which is equivalent to **minimizing the total operational cost** of the microgrid.

---

Out of the **25 possible scenarios** provided by **Pymgrid**, in this work we focus exclusively on scenarios **0** and **22** for training and analysis of the RL agents.  

| ID | Avg Load  | Avg PV   | Max Battery | Min Battery | Max Genset | Min Genset | Max Grid Import | Max Grid Export |
|----|-----------|----------|-------------|-------------|------------|------------|-----------------|-----------------|
| 0  | 483.87    | 160.37   | 1452.00     | 290.40      | -          | -          | 1920.00         | 1920.00         |
| 22 | 58584.64  | 18478.25 | 292924.00   | 58584.80    | 80372.70   | 4465.15    | 160744.00       | 160744.00       |

</br>

## :mag_right: Results and Discussion <a id="results-and-discussion"></a>

### Experimental setup
The experiments were run using the **discrete** Pymgrid environments to prioritize reproducibility and control compute cost. Each experiment followed the same pipeline: (1) scenario definition, (2) automated hyperparameter search (Ray Tune + Optuna with ASHA), and (3) evaluation of the best learned policy versus a rule-based controller (RBC). Training budgets and search spaces are described in the project report.

### Key results

#### Comparative results: DQN vs RBC

| Scenario   | Rule Based Control (RBC)     | DQN                            | Improvement (%) | Cost Benefit (€) |
|------------|------------------------------|--------------------------------|-----------------|------------------|
| **0**      | Total cost: 956,059.6622     | Total cost: 956,130.4997       | -0.007413 %     | -70.88 €         |
|            | Mean cost: 109.15            | Mean cost: 109.1597            |                 |                  |
| **22**     | Total cost: 44,142,821.0118  | Total cost: 44,142,821.0118    | ≈ 0 %           | ≈ 0 €            |
|            | Mean cost: 5,039.71          | Mean cost: 5,039.7101          |                 |                  |


#### Comparative results: PPO vs RBC


| Scenario   | Rule Based Control (RBC)     | PPO                            | Improvement (%) | Cost Benefit (€) |
|------------|------------------------------|--------------------------------|-----------------|------------------|
| **0**      | Total cost: 956,059.6622     | Total cost: 956,130.4997       | ≈ 0 %           | -0.34 €          |
|            | Mean cost: 109.15            | Mean cost: 109.15              |                 |                  |
| **22**     | Total cost: 44,142,821.0118  | Total cost: 44,142,821.0118    |   0 %           |  0 €             |
|            | Mean cost: 5,039.71          | Mean cost: 5,039.71            |                 |                  |


### Discussion / interpretation
1. **Learned policies mirror RBC behavior.** In multiple experiments the most frequently selected actions by RL agents matched the RBC’s priority list. That explains the parity in economic performance: the rule-based logic is already close to a locally optimal policy for the tested scenarios. This can be verified by comparing the energy flow graphs in the memory of the RBC algorithm with those of the DQN and PPO algorithms.

2. **Action space and problem formulation limit improvement.** When the discrete action set and reward design strongly reflect the RBC logic, agents tend to learn equivalent strategies rather than superior alternatives.

3. **Temporal dependencies may be underexploited.** Microgrids exhibit strong temporal dynamics. Using feed-forward approximators can miss useful temporal patterns — recurrent architectures (LSTM/GRU) or sequence-aware input representations may capture more long-term structure.

4. **Search depth and compute budget.** The explored hyperparameter ranges and training budgets were limited by available compute. Wider or longer searches might uncover configurations that outperform RBC.

### Limitations & recommended next steps
**Limitations**
- Experiments limited to discrete environments and a small set of scenarios.  
- Compute constraints limited the depth of hyperparameter search and training length.

**Recommendations**
1. **Try sequence models (LSTM/GRU):** in the policy/Q network to capture temporal dependencies.

2. **Expand hyperparameter search and increase training timesteps** 

3. **Experiment with hybrid approaches:** Experiment with **hybrid approaches**. The selected algorithms (DQN/PPO) may not be the most suitable for this specific problem. The original **Pymgrid paper** suggests that to achieve superior performance over the Rule-Based Control (RBC) method, a hybrid algorithm combining **Q-Learning with Decision Trees** was used. Exploring a similar combination (e.g., **DQN or PPO with a Decision Tree**) could be highly beneficial, as it would leverage the feature-learning capabilities of DRL with the explicit decision-making structure of a tree-based model.

</br>

## :bulb: Conclusion <a id="conclusion"></a>
For the tested scenarios and settings, **DQN and PPO did not produce economically significant improvements over the rule-based controller**: learned policies converged to behaviors similar to the RBC and produced comparable costs. To obtain practical improvements, future work should explore temporal architectures, broader hyperparameter searches, and combination of diferent arquitectures (hybrid approache) to improve performance.

</br>

## :link: References <a id="references"></a>

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller,M.(2013).Playing atari with deep reinforcement learning. https://doi.org/10.48550/arXiv.1312.5602

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., 60 Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533. https://doi.org/10.1038/nature14236

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv. https://doi.org/10.48550/arXiv.1707.06347

“The 37 Implementation Details of Proximal Policy Optimization,” ICLR Blog Track (implementation checklist), Mar. 2022. [Online]. Available: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Henri, G., Levent, T., Halev, A., Alami, R., & Cordier, P. (2020).pymgrid: An Open-Source Python Microgrid Simulator for Applied Artificial Intelligence Research. https://arxiv.org/abs/2011.08004
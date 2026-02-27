# Exploring Emergent Flocking Behavior: A Deep Reinforcement Learning Hypothesis

## Introduction
Starling murmurations are one of nature's most spectacular displays—thousands of birds moving in highly synchronized, fluid patterns that resemble a single, living entity. These complex aerial dances are not just visually stunning; they are hypothesized to be sophisticated survival mechanisms. By flocking closely, starlings optimize biological imperatives: confusing predators, facilitating social communication, and sharing aerodynamic effort.

For decades, computer scientists have sought to replicate these dynamics. Traditionally, this was achieved using static, rule-based systems like Craig Reynolds' seminal 1986 *Boids* algorithm. While beautifully mimicking the end result, Boids relies on hardcoded parameters for separation, alignment, and cohesion. In the natural world, birds don't follow these explicit equations; they adapt their behavior based on evolutionary and real-time survival pressures.

Our research shifts the paradigm from hardcoded rules to dynamic adaptation. By leveraging Multi-Agent Reinforcement Learning (MARL), we strip away explicit flocking instructions and instead equip artificial agents with basic biological senses and a singular objective: **survive**. This blogpost outlines the methodology of our custom simulation environment. We detail the strict mathematical foundation of our flight dynamics, perception, and predatory threats.

**A Note on Current Progress:** The fundamental goal of this exercise is to understand what motivates flocking behavior from first principles. We do not yet *know* that our agents will flock like starling murmurations, and we are not yet seeing full murmuration dynamically emerge in our early RL training runs. Our current methodology represents a hypothesis—that unified flocking is the thermodynamically and biologically optimal survival strategy under predator threat. As we analyze simulation results, our framework, rewards, and constraints will likely evolve.

## Background
The classic *Boids* model demonstrates how macroscopic flocking emerges from microscopic rules based on local perception. Each boid steers to avoid crowding (Separation), steers towards the average heading of flockmates (Alignment), and moves toward the average position of the flock (Cohesion). 

While the Boids algorithm successfully creates the illusion of flocking, it lacks algorithmic adaptability. Tuning these rules for different environments or introducing predatory threats requires manual recalibration. 

Reinforcement Learning (RL) addresses these limitations. In our approach, we use a decentralized-actor, centralized-critic architecture where agents *learn* how to move. Instead of being told *how* to act, agents are penalized for dying or colliding, and rewarded for staying alive. We hypothesize that the rules of Boids—separation, alignment, and cohesion—will emerge naturally as the statistically optimal strategy for survival.

## Methodology

To simulate and train these biologically inspired agents from scratch, we developed a highly optimized, custom RL environment. Our implementation relies on a fully vectorized PyTorch physics engine, enabling the simulation of hundreds of agents simultaneously on GPU infrastructure.

### The Physics Engine: Avian Dynamics and Updates
Unlike traditional grid-world or unconstrained continuous environments, our physics engine enforces strict biological flight constraints. Boids cannot stop, hover, or make instantaneous 180-degree turns.

At each timestep $\Delta t = 0.1$, the environment processes updates in the following highly specific manner:

28: 1.  **Force Application:** The neural network outputs an action vector $\mathbf{a}_i \in [-1, 1]^3$. This action explicitly maps to the 6-DOF controls of flight: Thrust, Roll Rate, and Pitch Rate.
29:     *   **Thrust:** The first dimension dictates forward acceleration along the bird's local forward axis, scaled by $F_{max}$.
30:     *   **Roll and Pitch:** The second and third dimensions apply angular rotation rates around the bird's local forward and right axes, respectively, allowing banking and diving.
31:     The environment computes these localized spherical rotations (e.g., Rodrigues' rotation formula) to maintain mathematically rigorous tracking of each agent's full 3D orientation (Forward, Right, Up) at all times.
32: 
33: 2.  **Kinematic Update:** We compute the new velocity based on Newtonian momentum and the applied Thrust force, updating the velocity vector dynamically:
34:     $$
35:     \vec{v}_{t+1} = \vec{v}_t + (Thrust \cdot \hat{v}_t) \Delta t
36:     $$
37: 
38: 3.  **Aerodynamic Drag and Constraints:** Real birds are limited by air resistance and physiological limits. We enforce a strict speed capacity, clipping the magnitude of the velocity vector between $v_{min} = 0.5$ and $v_{max} = 10.0$ to prevent infinite acceleration while preserving fluid momentum.
39: 
40: 4.  **Constant Position Update:** Finally, positions are updated natively via $\vec{x}_{t+1} = \vec{x}_t + \vec{v}_{t+1} \Delta t$. By stripping away unnatural static velocity assignments, agents must learn to regulate their own speed inline with their turning radii and survival needs.

### The Predator Mechanics: The Falcon State Machine
To introduce realistic, dynamic survival pressure, we designed "Falcon" predator agents governed by a robust, non-differentiable state machine. Predators fly faster ($1.5 \times v_{base}$) and turn tighter ($1.5 \times \theta_{max}$) than boids. Their behavior loops through four strict states:

1.  **VANTAGE (0):** The predator spawns and flies upwards to a height of $z \ge 0.9L$ (where $L$ is the boundary length). Upon reaching the vantage altitude, it holds its altitude and waits for a fixed `reset_duration` (50 timesteps) to survey the flock before transitioning.
2.  **LOITER (3):** The predator wanders randomly above the flock at 30% of its top speed for a highly variant duration ($100 + \mathcal{U}(0, 300)$ timesteps). During this phase, it constantly evaluates the positional structure of the flock below.
3.  **Target Selection & Evaluation:** At every single timestep during the HUNTING and DIVING states, the predator evaluates the flock to dynamically update its target $\vec{x}_{tgt}$.
    *   **Isolation Metric:** For every alive boid $i$, the environment calculates a neighborhood density: $N_i = \sum_{j \neq i} \mathbb{I}(|\vec{x}_i - \vec{x}_j| < r_{percept})$. A boid is considered mathematically "isolated" if $N_i < N_{min}$ (where $N_{min} = 5$).
    *   **State Choice:** When LOITER expires, if $\exists i$ such that $N_i < 5$, the predator enters **HUNTING**. Otherwise, it enters **DIVING**.
4.  **HUNTING vs DIVING Execution:** 
    *   *Hunting (Targeting Isolation):* The predator calculates the distance to all currently isolated boids: $d_{pj} = |\vec{x}_p - \vec{x}_j|$. At *every timestep*, it dynamically asserts its $\vec{x}_{tgt}$ to the coordinate of the isolated boid where $d_{pj}$ is minimized. If the boid rejoins the flock, the predator will instantly snap to the next closest isolated boid.
    *   *Diving (Targeting Density):* If no boids are isolated, the predator executes a high-speed plunge to scatter the flock. It finds the absolute closest boid $k$, calculates the subset of boids $V_k$ within $k$'s perception radius, and sets its $\vec{x}_{tgt}$ to the mathematical Center of Mass of that local sub-flock: $\vec{x}_{tgt} = \frac{1}{|V_k|} \sum_{v \in V_k} \vec{x}_v$.
    *   *Dive Completion:* A dive is considered structurally complete when the predator's altitude drops past the target ($z_p < z_{tgt} + 5.0$) while maintaining a negative z-velocity ($v_{pz} < 0$).
5.  **Capture Mechanics:** At every timestep, the environment computes the 2D pairwise distance matrix between all predators and all boids. If any distance $|\vec{x}_p - \vec{x}_i| < r_{catch}$ (where $r_{catch} = 2.0$), boid $i$ is killed. The successful predator instantaneously resets to **VANTAGE**, computing a new random $x,y$ coordinate near the center $L/2$ to climb back toward.

**Crucially, the predator does *not* explicitly evaluate the probabilistic "likelihood of capture"** or perform complex path planning (like pure pursuit interception or calculating optimal intercept angles). It utilizes a greedy, instantaneous sensory baseline: steering blindly toward the current coordinate of the closest isolated boid, or the current center-of-mass of the closest dense sub-flock.

### Observation Space: Biological Perception
To make decisions, each starling $i$ processes a strictly localized 18-dimensional continuous observation vector. This simulates limited biological senses over a `perception_radius` of 15.0 units.

*   **Proprioception:**
    *   Normalized 3D velocity: $\vec{v}_i / v_{base}$
    *   3D spatial bounds: 3D relative position mapped natively between $[-1.0, 1.0]$.
*   **Group Context:**
    *   *Nearest Neighbor Distance ($d_{min}$):* Normalized by the perception radius, $d_{min} / r_{percept}$. If no neighbors are seen, it defaults to $1.0$.
    *   *Local Density ($\rho$):* The true fraction of the total population currently within the perception zone.
    *   *Local Alignment:* The unit vector of the average velocities of all visible neighbors: $\frac{1}{|N_i|} \sum_{j \in N_i} \vec{v}_j$, subsequently unit-normalized.
    *   *Center of Mass (CoM) Direction:* The unit vector pointing from $\vec{x}_i$ to the mean positional coordinate of visible neighbors.
*   **Perceptual Threat Matrix:** We designed specific visual proxies for escaping predators:
    *   Let $\vec{x}_p$ be the position of the closest predator, with distance $d = |\vec{x}_p - \vec{x}_i|$.
    *   Let $\vec{u} = (\vec{x}_p - \vec{x}_i)/d$ be the unit bearing pointing towards the predator.
    *   *Distance:* Normalized distance limited by environmental space, $d / (L/2)$.
    *   *Closing Speed ($v_{close}$):* The scalar rate at which the predator is approaching: $v_{close} = - (\vec{v}_p - \vec{v}_i) \cdot \vec{u}$. This is normalized to $[-1, 1]$ relative to the absolute maximum closure rate $v_{pred} + v_{base}$.
    *   *Looming (Time-to-Collision Proxy):* The rate of optical expansion of the predator on the retina, mathematically formulated as $\text{loom} = v_{close} / \max(d, \epsilon)$. It is strictly clamped to prevent numerical explosions when $d \to 0$.
    *   *Bearing (In Front?):* The dot product of the boid's heading and the predator's bearing, $\hat{v}_i \cdot \vec{u}$.

### Reward Function: Structure over Strategy
The core philosophy of our approach is that **flocking is not explicitly rewarded**. The reward function attempts to reflect pure structural and biological endpoints:
*   **Survival:** $+0.1$ for every frame survived without capture.
*   **Death Penalty (Predator or Bounds):** First-time transition to the "dead" state via predator capture incurs a stark terminal $-100.0$ penalty. Furthermore, we enforce **Purist RL Constraints**: there are no invisible physical walls pushing boids back inside the environment. If a boid flies strictly out of bounds ($x,y,z \notin [0, L]$), it is considered a fatal crash. The agent's episode terminates, and it immediately receives the same massive $-100.0$ death penalty.
*   **Collision Penalty:** To implicitly enforce the instinctual "Separation" rule, collisions with peers ($d < 1.0$) incur a $-2.0$ penalty per peer per frame.
*   **Boundary Warning:** Penetrating the soft boundary margin ($10\%$ of space length) incurs a scaling geometric penalty up to $-5.0$ proportional to boundary penetration, providing a psychological deterrent before the fatal crash at $0$.

### Neural Architecture: MAPPO
We utilize Multi-Agent Proximal Policy Optimization (MAPPO), an actor-critic algorithm highly suited for multi-agent swarm environments.
*   **Shared Actor Network:** A multi-layer perceptron (two hidden layers of 64 units with Tanh activations) processes the 18D local observation space to output a probability distribution over the continuous 3D action space. By sharing the actor network weights, all agents learn a unified, symmetrical behavioral policy.
*   **Centralized Critic:** To resolve the massive credit assignment problem in a dense swarm, our Critic network (256 hidden units) possesses total omniscience during training. It evaluates the concatenated global observation state of all entities to guide the Actor's policy updates. During simulation (inference), the Critic is discarded, and agents act purely on decentralized local perception.

## Results

Below is an early visualization of our 3D Starling and Falcon simulation environment:

![Murmuration RL Simulation](assets/murmuration_rl.gif)

*Placeholder for extended Results and Visualizations. As robust simulation models progress, we will document whether cohesive murmurations mathematically emerge as the optimal policy to defeat the Falcon state machine, or if the agents evolve unexpected alternative survival paradigms.*

## Conclusion
By bridging biological imperatives with deep reinforcement learning, our environment establishes a rigorous testbed for exploring evolutionary survival tactics. While traditional Boids prove that local rules *can* create flocking, our simulation will reveal whether local rules *must* create flocking under threat. If driven solely by the desire to survive predatory attacks, we hypothesize that these artificial agents will organically discover that numerical and geographic unity is strength.

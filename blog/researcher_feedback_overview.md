# Murmur RL: Code-Grounded Research Overview

## Purpose of this document

This note reconstructs what the repository currently implements, with emphasis on the environment setup and the learning methodology. It is written for senior RL researchers who want to review the experimental design, identify hidden inductive biases, and suggest improvements before stronger claims are made about emergent flocking.

Two important framing points:

1. This document is based on the current code and test suite, not just the narrative in `blog/blogpost.md`.
2. Several public-facing descriptions in the repo are now stale. In particular, the current training stack is a learned predator-prey co-evolution system, not merely prey agents against a hand-coded predator state machine.

## Executive Summary

At its core, this repo is a continuous-control multi-agent predator-prey testbed in 3D. A population of "starlings" learns a shared prey policy; a smaller population of "falcons" learns a shared predator policy. The prey receive local biological-style observations, the predators receive a different observation model with density-dependent visual corruption, and both populations are trained under a centralized-training / decentralized-execution regime using PPO-style actor-critic updates.

The most important technical reality is that the present system is not a "pure survival objective with no flocking bias." The implementation currently includes at least two direct pro-grouping inductive biases:

- A density-based potential term in the prey reward.
- Predator observation corruption whose magnitude increases with target local density.

So the current scientific question is better phrased as:

> Under soft boundary shaping, collision penalties, density shaping, and density-dependent predator confusion, does coordinated flocking emerge as an effective prey policy under co-evolutionary pressure?

That is still interesting, but it is more structured than a first-principles "survival alone" story.

## Repository Reality Check

### What is currently authoritative

The operative code paths are:

- `src/murmur_rl/envs/physics.py`
- `src/murmur_rl/envs/vector_env.py`
- `src/murmur_rl/agents/starling.py`
- `src/murmur_rl/training/ppo.py`
- `src/murmur_rl/training/runner.py`
- `simulate.py`

The legacy reference environment in `src/murmur_rl/envs/murmuration.py` is still useful, but it is mainly a PettingZoo-style correctness baseline for the newer tensorized environment.

### What appears stale or partially stale

- `README.md` still describes predators as a "non-differentiable predatory state-machine"; the active training setup uses learned predator policies.
- `blog/blogpost.md` describes a related but older conceptual design. Several implementation details have drifted, including frame stack size, critic construction, and the exact predator training loop.

## Reproducibility and Environment Setup

### Software stack

- Python requirement: `>=3.12`
- Package management: `uv`
- Core dependencies:
  - `torch>=2.10.0`
  - `gymnasium>=1.2.3`
  - `pettingzoo>=1.25.0`
  - `numpy>=2.4.2`
  - `matplotlib>=3.10.8`
  - `imageio`, `imageio-ffmpeg`
  - `wandb>=0.25.0`
- Dev dependency: `pytest>=9.0.2`

The project is laid out as a `src/` package. In practice, running code from the repo root currently benefits from either:

- installing the package in editable mode, or
- setting `PYTHONPATH=src`

I verified the test suite locally with:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest tests/test_vector_env.py tests/test_naive_validation.py -q
```

Result: `41 passed`.

### Device strategy

Training selects the device in this order:

- `mps` if available
- else `cuda` if available
- else `cpu`

The environment and networks optionally use `torch.compile`, with runtime guards around Triton availability and backend support. The design intent is clear: keep the full rollout path on-device and avoid Python/dictionary overhead in the main training loop.

### Operational entry points

The current operational scripts are:

- Training: `src/murmur_rl/training/runner.py`
- Visualization / rollout playback: `simulate.py`

A practical training invocation is:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m murmur_rl.training.runner --checkpoints-dir checkpoints
```

Resume is supported through the prey checkpoint path, with the predator checkpoint inferred by filename substitution:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m murmur_rl.training.runner --resume checkpoints\starling_brain_ep500.pth
```

Visualization is driven through:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe simulate.py --checkpoint checkpoints\starling_brain_ep500.pth --frames 1800 --num-boids 250 --num-predators 4
```

## Environment Setup

### 1. World Geometry and Episode Structure

The active environment is `VectorMurmurationEnv`, a fully tensorized 3D continuous environment.

Default training configuration in `runner.py`:

| Parameter | Value |
| --- | --- |
| Number of prey agents | 100 |
| Number of predators | 10 |
| Space size | 50.0 |
| Perception radius | 15.0 |
| Episode horizon | 500 steps |
| Time step `dt` | 0.1 |

Important implementation detail:

- The world is a nominal cube, but the boundaries are soft, not hard.
- There is no position clipping, no bounce, and no terminal out-of-bounds condition in the active vectorized environment.
- Instead, agents are pulled inward only through a boundary potential in the reward.

So the environment is best understood as a soft-constrained 3D space with a center-seeking shaping term, not a hard-walled aviary.

### 2. Spawn Initialization

### Prey initialization

At reset:

- Boid positions are sampled uniformly inside a centered spawn box of side length `min(100, space_size)`.
- Since training uses `space_size=50`, this means prey are initially spread across the full 50-unit cube.
- Initial prey velocities are random directions normalized to `base_speed`.
- An orthonormal flight frame is constructed using the velocity vector and an `up_vector`.

### Predator initialization

Predators are initialized separately:

- Positions are random in the full cube.
- Then one coordinate axis is snapped to either `0` or `space_size`, so predators start on a boundary face.
- Initial predator velocities are random directions at predator base speed.
- Predators also maintain an `up_vector` for 3D orientation updates.

This setup creates a broad initial separation between an interior swarm and edge-injected hunters.

### 3. Flight Dynamics

The physics engine is in `src/murmur_rl/envs/physics.py`.

### Prey controls

Each prey action is 3D:

- `a[0]`: thrust
- `a[1]`: roll
- `a[2]`: pitch

The dynamics are not position-jump controls. They are closer to simplified 6-DOF flight kinematics:

- roll rotates the `up/right` frame around forward
- pitch rotates the forward/up frame around right
- speed is then updated via thrust
- position advances via `x_{t+1} = x_t + v_{t+1} dt`

Key prey physics constants under the default training config:

| Parameter | Value |
| --- | --- |
| `base_speed` | 5.0 |
| `min_speed` | 2.5 |
| `max_turn_angle` | 0.5 rad / step |
| `max_force` | 2.0 |
| `dt` | 0.1 |

Speed update:

- `new_speed = old_speed + thrust * dt`
- then clamped to `[min_speed, base_speed]`

So the prey cannot stop or hover. They always move, and braking only reduces speed to the minimum flight speed floor.

### Predator controls

Each predator action is also 3D:

- `a[0]`: sprint intent
- `a[1]`: roll
- `a[2]`: pitch

Predators share the same orientation update structure, but they differ in three ways:

- higher sprint speed
- higher turn authority
- stamina and cooldown mechanics

Predator parameters derived from prey parameters:

| Parameter | Value |
| --- | --- |
| Predator base speed | `base_speed` |
| Predator sprint speed | `1.5 * base_speed` |
| Predator turn angle | `1.5 * max_turn_angle` |
| Initial catch radius | `2.0` |

### Stamina economy

Predators have:

- `predator_max_stamina = 100`
- `predator_sprint_drain = 1.0` per step
- `predator_recovery_rate = 0.5` per step
- `predator_cooldown_duration = 50` steps after a successful catch

Predator speed is not switched instantaneously. It is inertia-smoothed:

- `new_speed = 0.9 * current_speed + 0.1 * target_speed`

During cooldown after a catch, predators are forced to a reduced speed of `0.5 * predator_base_speed`.

### Capture mechanics

A prey agent dies if any predator is within `predator_catch_radius`.

The vectorized environment then persists prey death through a boolean `_dead_mask`. Dead prey:

- keep zero velocity
- stop moving
- remain part of fixed-size tensors

This is a standard and efficient way to preserve shape consistency for batched PPO.

### 4. Curriculum

The active environment includes a built-in curriculum on predator lethality:

- predator catch radius starts at `2.0`
- it decays linearly to `0.5`
- decay schedule: `5,000,000` environment steps

This is a meaningful design choice. Early training uses forgiving capture geometry, while later training demands tighter predator-prey interception accuracy.

### 5. Observation Design

### Prey local observation: 18 dimensions

The prey observation vector is:

| Block | Dim | Description |
| --- | --- | --- |
| Own normalized velocity | 3 | velocity divided by prey base speed |
| Nearest-neighbor distance | 1 | nearest alive prey distance / perception radius |
| Local density | 1 | count of neighbors within radius / total prey count |
| Local alignment | 3 | normalized average neighbor velocity |
| Local center-of-mass direction | 3 | normalized direction toward local neighbor COM |
| Predator distance | 1 | closest predator distance / half-space |
| Predator closing speed | 1 | normalized relative closure rate |
| Looming proxy | 1 | `v_close / d`, clipped |
| Predator bearing | 1 | dot between prey velocity direction and predator bearing |
| Relative position to environment center | 3 | `(x - center) / half_space` |

Comments:

- This observation is strongly "boids flavored" even though classic alignment and cohesion are not written directly as control rules.
- Alignment and COM direction are explicitly supplied as features.
- The prey therefore do not need to infer local group structure from raw pairwise geometry alone.

### Predator local observation: 45 dimensions

Predator observations are richer and asymmetric:

| Block | Dim | Description |
| --- | --- | --- |
| Relative position in world | 3 | predator position relative to world center |
| Normalized predator velocity | 3 | scaled by sprint speed |
| Normalized stamina | 1 | current stamina / max stamina |
| Relative COM of alive prey swarm | 3 | center of mass of alive prey relative to predator |
| Five nearest target bundles | 35 | each target contributes 7 dims |

Each of the five target bundles contains:

- noisy relative target position: 3
- relative target velocity: 3
- normalized target distance: 1

### Visual obfuscation mechanism

For each observed target, the predator receives a noisy target position:

- compute the target prey's local density
- define `sigma = density * 5.0`
- sample Gaussian noise with that standard deviation
- add noise to target position before relative encoding

This is one of the repo's most consequential modeling choices. It explicitly operationalizes predator confusion as an engineered perceptual channel, rather than letting confusion emerge only from geometry and occlusion. This almost certainly changes the equilibrium structure of the game.

### 6. Global State for the Centralized Critic

Despite some language elsewhere suggesting a mean-field critic, the active implementation is not mean-field in the usual MARL sense.

The centralized critic state is built via:

- the focal agent's own local observation
- exact relative state of the `K=min(10, N)` nearest prey agents
- exact relative state of all predators

For each of the `K` nearest prey, the critic gets:

- relative position: 3
- normalized velocity: 3
- alive flag: 1

For each predator, the critic gets:

- relative position: 3
- normalized velocity: 3

Therefore:

- prey critic input dimension = `18 + 10*7 + P*6`
- with `P=10`, that becomes `18 + 70 + 60 = 148`
- predator critic input dimension = `45 + 10*7 + P*6`
- with `P=10`, that becomes `45 + 70 + 60 = 175`

This is better described as a structured local-relational critic than a mean-field critic.

## Reward Design

### 1. Prey reward

The prey reward has four active ingredients.

### Survival reward

- `+0.1` each step while alive

### Collision penalty

- `-2.0` per prey-prey collision
- a collision is defined as pairwise distance `< 2.0`

This is a strong short-range anti-collapse term.

### Death penalty

- newly killed prey receive `-100.0`
- already dead prey receive `0`

### Potential-based shaping

The environment maintains a potential:

- boundary term: `phi_bounds = -k * ||relative_position_from_center||^2`
- density term: `phi_density = c * local_density`
- total potential: `phi = phi_bounds + phi_density`

The shaping contribution is:

`gamma * phi(s') - phi(s)`

This is textbook PBRS in form, though the scientific interpretation matters. The density term explicitly rewards being in locally dense regions. That means the current setup does not merely allow flocking to emerge; it rewards movement toward higher social density.

### 2. Predator reward

Predator reward includes:

- `+10.0` per newly killed prey assigned to that predator on the current step
- `-0.05` hunger penalty each step when not in cooldown and not making a catch
- predator boundary potential shaping analogous to the prey boundary term

This makes predators pressure the swarm continuously rather than loitering after initial failures.

### 3. No hard boundary death

Earlier narratives mention edge pathologies and out-of-bounds issues. In the current active environment:

- there is no terminal boundary death
- there is only a center-seeking quadratic potential

That is the right description to carry into any researcher discussion.

## Solution Methodology

### 1. Problem Formulation

The training setup is a two-population continuous-action Markov game:

- population 1: starlings
- population 2: falcons

Each population shares parameters internally:

- one prey actor-critic shared across all prey
- one predator actor-critic shared across all predators

This is a standard population-sharing choice for symmetric multi-agent settings and greatly reduces the effective parameter count.

### 2. CTDE Structure

Execution is decentralized:

- prey actors use only prey local stacked observations
- predator actors use only predator local stacked observations

Training is centralized at the critic:

- each population uses a critic that consumes stacked global state

So the setup is unequivocally CTDE, implemented in a PPO-style optimization loop.

### 3. Temporal Context

The active code uses `stacked_frames = 4`, not 3.

For both prey and predators:

- the current observation is repeated into a 4-frame buffer at reset
- each step appends the newest frame and drops the oldest
- dead prey have their history buffer reset to repeated copies of the newest post-death observation

This is a simple but reasonable way to inject short-horizon motion information without recurrent networks.

### 4. Network Architecture

### Prey network

`StarlingBrain` uses:

- actor feature extractor: two hidden layers of size 128 with `Tanh`
- actor mean head: one hidden layer of size 128 plus `Tanh` output
- a learned global log-standard-deviation parameter for Gaussian exploration
- critic: two hidden layers of size 512 with scalar output

### Predator network

`FalconBrain` is similar but wider on the actor:

- actor feature extractor hidden size 256
- actor mean head hidden size 256
- learned Gaussian log-standard-deviation
- critic hidden size 512

This asymmetry is intentional and sensible: the predator observation model is larger and noisier than the prey observation model.

### 5. PPO Training Loop

The trainer in `src/murmur_rl/training/ppo.py` runs as follows:

1. Reset environment and initialize frame stacks.
2. Collect a fixed rollout of `500` steps for both populations simultaneously.
3. Store local observations, centralized critic observations, actions, log-probs, rewards, dones, and values.
4. Compute GAE advantages and returns separately for prey and predators.
5. Normalize value targets with separate running mean / variance objects for the two populations.
6. Optimize each population independently using clipped PPO losses.

Active hyperparameters in `runner.py`:

| Hyperparameter | Value |
| --- | --- |
| Rollout length | 500 |
| Training epochs | 30000 |
| Prey actor LR | `3e-4` |
| Predator actor LR | `1e-4` |
| Critic LR | `1e-3` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| PPO clip | `0.2` |
| Entropy coefficient | `0.01` initially |
| Value coefficient | `0.5` |
| Max grad norm | `0.5` |
| PPO update epochs | `4` |
| Batch size | `1024` |
| Target KL | `0.015` |

The entropy coefficient decays linearly toward zero over roughly the first `1000` epochs, which acts as a simple exploration-to-consolidation schedule.

### 6. Logging and Checkpointing

The runner logs:

- policy loss
- value loss
- entropy
- explained variance
- mean GAE return

for both prey and predators.

It also logs two domain-specific proxy metrics:

- mean predator distance
- mean local social neighbors

Checkpoints are written every `500` epochs as paired files:

- `starling_brain_ep{epoch}.pth`
- `falcon_brain_ep{epoch}.pth`

## What Is Actually Strong About This Setup

Several design choices are solid and worth preserving:

- The core environment is fully tensorized and avoids Python-level per-agent loops on the hot path.
- There is a careful legacy-to-vectorized validation story.
- The local observation design is compact rather than fully global.
- The predator-prey co-evolution setup is more interesting than a fixed scripted predator.
- Frame stacking is a pragmatic way to expose short-range motion cues without RNN complexity.
- The K-nearest critic representation is much more scalable than full concatenation of every entity state.

Most importantly, the repo already includes a real correctness culture. The validation suite compares:

- vectorized environment vs legacy environment
- vectorized logic vs naive reference implementations
- multi-step invariants
- observation and reward shapes
- flight orientation orthonormality
- stamina and cooldown logic
- global-state construction

That is better than what many research prototypes start with.

## High-Priority Research and Engineering Questions

These are the issues I would put in front of senior RL reviewers first.

### 1. The current system is not testing "survival alone"

This is the most important conceptual point.

The prey are not only rewarded for survival. They also receive:

- explicit density-based PBRS
- local alignment and COM direction features
- an environment where predator sensing becomes noisier exactly when prey are locally dense

So the current study is not a minimal emergence test. It is a shaped emergence test with strong social inductive biases. That may be fine, but it should be stated explicitly.

### 2. Training-time action support does not match the claimed bounded action space

This is the most important implementation issue.

The actors output:

- Gaussian means squashed by `Tanh`
- but sampled actions come from `Normal(mean, std)` with no tanh-squash and no clamp before entering physics

Consequences:

- during training, actions can exceed `[-1, 1]`
- the physics layer then scales those out-of-range values directly
- during evaluation in `simulate.py`, the code uses deterministic actor means, which are bounded by `Tanh`

So the train-time and eval-time control distributions are not the same. This is a serious methodological mismatch for a continuous-control paper. If bounded actions are intended, the standard fix is a squashed Gaussian policy with the corresponding corrected log-prob, or at minimum an explicit clamp with awareness of the bias it introduces.

### 3. Predator kill credit may be over-assigned

Predator reward attribution is reconstructed from a fresh predator-boid distance check after deaths are marked. If multiple predators are within catch radius of the same newly dead prey, more than one predator may receive reward for the same kill.

That may or may not be common in practice, but it is worth auditing because it changes predator credit assignment and may encourage pack-overlap behaviors that are artifacts of the implementation.

### 4. The "space" is soft, not bounded

The comments sometimes suggest normalized positions in `[-1, 1]`, but there is no hard position constraint. If an agent leaves the nominal cube, position-derived features can exceed that range. This does not break training, but it matters when interpreting the environment and when comparing to the narrative of a confined volume.

### 5. The critic is not mean-field

If the paper or blog says "mean-field MAPPO," that should be revised. The implemented critic is a K-nearest relational critic with all predator states appended, not a mean-field summary over neighboring agents.

### 6. Evaluation metrics are still weak relative to the emergence claim

The logged biology metrics are useful, but they are not yet enough to support a strong claim about murmuration-like flocking.

At minimum, I would want:

- nearest-neighbor distance distribution
- polarization / heading consensus
- angular momentum or toroidal motion metrics
- connected-component statistics under distance graphs
- predator capture rate over time
- swarm spatial entropy
- attack success by target rank from the swarm edge

Without those, videos may look compelling while the underlying collective structure remains ambiguous.

### 7. Reproducibility controls are not yet first-class

The repo has tests, which is excellent, but experiment reproducibility is still fairly lightweight. I did not see strong experiment management for:

- seeds
- versioned configs
- exact train/eval protocol snapshots
- checkpoint metadata beyond filename conventions

For a research feedback round, I would flag this as a process improvement area rather than a defect.

## Recommended Reframing for External Review

If this document is being sent to experienced RL researchers, I would present the project as follows:

> We built a tensorized 3D predator-prey MARL environment in which prey and predators are trained via CTDE PPO with shared population policies. The prey receive local social and predator-threat observations, predators receive density-corrupted target observations, and both populations operate under biologically inspired flight constraints. We are investigating whether stable flock-like collective strategies emerge under this game, while recognizing that the current environment includes explicit pro-grouping inductive biases through density shaping and predator confusion modeling.

That framing is accurate, technically defensible, and still scientifically interesting.

## Suggested Next Experiments

If the goal is to improve the scientific process rather than merely improve training curves, I would prioritize the following ablations.

### Ablation set A: remove inductive biases one by one

- remove density PBRS while keeping predator visual confusion
- remove predator visual confusion while keeping density PBRS
- remove both and test whether any flocking-like structure survives
- remove alignment and COM direction features from prey observations

This will tell you what is actually doing the heavy lifting.

### Ablation set B: fix action support

- replace the unsquashed Gaussian with a tanh-squashed Gaussian policy
- rerun a small benchmark and compare stability, entropy, and emergent behavior

### Ablation set C: predator reward attribution

- assign kill credit to the nearest predator only
- alternatively, track the actual catching predator in physics and pass that identity cleanly into reward computation

### Ablation set D: evaluation metrics

- log polarization, pair-correlation, density histogram, and capture-rate metrics every checkpoint
- compare random policy, early training, mid training, and late training

### Ablation set E: curriculum sensitivity

- no catch-radius annealing
- faster annealing
- slower annealing

This will clarify whether the emergent prey organization depends critically on curriculum shaping.

## Bottom Line

This repo already contains a credible and fairly sophisticated RL systems prototype:

- tensorized continuous 3D dynamics
- asymmetric predator-prey co-evolution
- CTDE PPO
- structured local observations
- explicit validation tests

The main scientific caveat is that the current environment meaningfully shapes prey social behavior instead of testing survival in a minimally biased setting. The main technical caveat is the mismatch between the claimed bounded action space and the unbounded Gaussian samples used during training.

If those two points are made explicit, this is absolutely worth circulating to senior RL researchers for feedback.

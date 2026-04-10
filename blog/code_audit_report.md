# Murmur RL Code Audit Report

## Scope

This audit focuses on the active code path used for training and simulation:

- `src/murmur_rl/envs/physics.py`
- `src/murmur_rl/envs/vector_env.py`
- `src/murmur_rl/agents/starling.py`
- `src/murmur_rl/training/ppo.py`
- `src/murmur_rl/training/runner.py`
- `simulate.py`

The goal was to find correctness issues and methodology hazards in the current implementation, not to re-state the project overview.

## Audit Method

I used three sources of evidence:

1. Static review of the active training and simulation code paths.
2. Full local test run:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q
```

Result: `46 passed`.

3. Targeted probe scripts to test hypotheses that are not covered by the existing suite:

- sampled-action bounds and train/eval mismatch
- duplicate predator reward assignment
- post-mortem collision and cooldown behavior
- critic-state edge cases
- tiny rollout/train smoke test

The small rollout/train smoke test completed successfully on CPU without W&B.

## Findings

### P1. PPO samples are not bounded to the documented action range, creating a train/eval mismatch

**Code**

- `src/murmur_rl/agents/starling.py:23-31`
- `src/murmur_rl/agents/starling.py:57-63`
- `src/murmur_rl/agents/starling.py:90-98`
- `src/murmur_rl/agents/starling.py:119-125`
- `src/murmur_rl/envs/physics.py:123-130`
- `src/murmur_rl/envs/physics.py:186-191`
- `simulate.py:147-153`

**What is happening**

The actor mean is squashed with `Tanh`, but the actual training action is sampled from an unconstrained `Normal(mean, std)`:

- training uses `probs.sample()`
- the sampled action is passed straight into physics
- physics assumes the action is already in `[-1, 1]`

At evaluation time, `simulate.py` does not sample from the Gaussian. It feeds the deterministic `actor_mean` output directly into the environment. That means:

- training sees unbounded actions
- evaluation sees bounded actions

So the train-time and inference-time control laws are not the same.

**Why this matters**

- The environment scales thrust, roll, and pitch as if the action were bounded.
- Out-of-range samples directly increase turn angles and thrust magnitude beyond the nominal control envelope.
- PPO is therefore optimizing under a different dynamics regime than the one used in simulation playback.

**Probe result**

Using the current policy head at initialization:

- `67.1%` of action vectors had at least one component outside `[-1, 1]`
- `31.26%` of scalar action components were outside bounds
- the largest sampled absolute action in a 10k sample was `4.2001`

**Recommended fix**

Use a tanh-squashed Gaussian policy and compute the corrected log-prob for PPO. If you do not want to implement the Jacobian correction immediately, an intermediate containment fix is to clamp sampled actions before they reach physics and explicitly note the bias this introduces.

### P1. Dead prey can re-trigger predator cooldown indefinitely

**Code**

- `src/murmur_rl/envs/physics.py:251-273`

**What is happening**

`_check_captures()` computes:

- `caught_matrix = dist_to_predator < predator_catch_radius`
- `caught_by_predator = caught_matrix.any(dim=1)`

This is done over all prey positions, including already-dead prey. Because dead prey remain frozen in the world, a predator that stays near a corpse can keep satisfying the catch condition and have its cooldown reset repeatedly.

**Why this matters**

- A corpse can act like an invisible cooldown mine.
- Predators can remain suppressed near a kill site even when no new prey are being captured.
- This changes the predator dynamics around successful attacks and can distort emergent behavior.

**Probe result**

With one predator and one prey co-located:

- after the first capture, cooldown became `50`
- calling `_check_captures()` again with the corpse still in place left cooldown at `50` instead of allowing it to decay

In a real environment step, the cooldown is decremented first and then reset again on the corpse, so the net effect is persistent retriggering while the predator remains nearby.

**Recommended fix**

Mask out dead prey before computing `caught_matrix`, e.g. only allow currently alive prey to trigger capture logic and predator cooldown.

### P1. The same prey kill can be rewarded to multiple predators

**Code**

- `src/murmur_rl/envs/physics.py:256-267`
- `src/murmur_rl/envs/vector_env.py:473-480`

**What is happening**

The environment marks a prey as dead if any predator is within catch radius. Later, predator rewards are reconstructed by checking which predators are within catch radius of the newly dead prey. If multiple predators are close enough, each of them receives full kill credit.

**Why this matters**

- Predator reward no longer reflects unique capture count.
- Pack overlap can be over-rewarded.
- Reported predator returns can exceed the number of actual prey deaths.

**Probe result**

With one prey and two predators at the same position:

- `new_deaths = [True]`
- predator rewards were `[10.0, 10.0]`
- total predator reward for a single kill was `20.0`

**Recommended fix**

Track the responsible predator explicitly in physics, or deterministically assign credit to the nearest predator at the moment of capture.

### P2. Dead prey still count for collision penalties, creating invisible obstacles

**Code**

- `src/murmur_rl/envs/vector_env.py:417-454`

**What is happening**

Collision penalties are computed from the full prey-prey distance matrix without masking dead prey. Because dead prey remain in the position tensor, live prey can be penalized for being near a corpse even though dead prey are visually removed from the rendered swarm.

**Why this matters**

- Live prey can be punished for proximity to invisible objects.
- The reward function mixes "separation from flockmates" with "avoid dead bodies," even though the latter is not documented as part of the task.

**Probe result**

With one dead prey and one live prey placed within the collision radius:

- the dead prey reward was `0.0`
- the live prey reward was `-1.9`, i.e. `+0.1` survival minus one collision penalty

**Recommended fix**

If corpses are meant to be removed from the social system, mask dead prey out of the collision matrix. If corpse avoidance is intentional, it should be made explicit in the environment description and visualized.

### P2. `get_global_state()` uses a batch-size heuristic that breaks when `num_agents == num_predators`

**Code**

- `src/murmur_rl/envs/vector_env.py:344-360`

**What is happening**

The function decides whether it is building boid or predator critic state via:

```python
batch_size = local_obs.shape[0]
is_boid = (batch_size == self.n_agents)
```

This is brittle. If `num_agents == num_predators`, predator observations are misclassified as boid observations and the wrong focal positions are used.

**Why this matters**

- The API silently returns semantically wrong critic inputs under a plausible configuration.
- The failure is not shape-based, so it is easy to miss.

**Probe result**

With `num_agents = num_predators = 3`, the returned predator global state differed from the intended predator-branch computation with max absolute error `1.7733`.

**Recommended fix**

Split this into two explicit methods, e.g. `get_boid_global_state()` and `get_predator_global_state()`, or add an explicit `agent_type` argument.

### P2. Invalid K-nearest critic slots are not zeroed; only the alive flag is zeroed

**Code**

- `src/murmur_rl/envs/vector_env.py:370-388`

**What is happening**

When there are fewer live prey than `K`, `topk` still returns indices for invalid slots. The code sets the final `alive` feature to zero for those slots, but it does not zero the associated relative positions or velocities.

So the critic can see:

- nonzero relative position
- nonzero velocity
- `alive = 0`

for padding / dead slots.

**Why this matters**

- Late-episode critic inputs contain arbitrary geometry from dead prey.
- The critic can learn to condition on garbage structure unless it perfectly learns to ignore those slots from the alive bit alone.

**Probe result**

With only one live boid remaining, the extra neighbor slots in the critic state had:

- `alive = 0`
- but visibly nonzero relative positions and velocities

**Recommended fix**

Multiply `rel_pos` and `rel_vel` by `k_alive` before concatenation so invalid slots are fully zeroed.

### P3. `pred_global_obs_dim` is wrong when there are fewer than five prey

**Code**

- `src/murmur_rl/envs/vector_env.py:54-55`
- `src/murmur_rl/envs/vector_env.py:302-341`

**What is happening**

The predator observation builder uses `k = min(5, self.n_agents)`, so the local predator observation shrinks when there are fewer than five prey. But `pred_global_obs_dim` is hardcoded as if predator local observations are always 45-D.

This is correct only when `num_agents >= 5`.

**Why this matters**

- The environment metadata is wrong for small swarm configurations.
- Code that trusts `pred_global_obs_dim` instead of the actual tensor shape will be brittle.

**Probe result**

Examples:

- `num_agents = 3`: actual predator global-state dim `64`, stored field `78`
- `num_agents = 4`: actual predator global-state dim `78`, stored field `85`

**Recommended fix**

Compute the field from the actual local predator observation width:

- local predator obs dim = `10 + min(5, n_agents) * 7`
- predator global dim = that local dim plus `K*7 + P*6`

### P3. Training config overrides do not fully propagate through derived predator physics parameters

**Code**

- `src/murmur_rl/training/runner.py:78-81`
- `src/murmur_rl/envs/physics.py:35-44`

**What is happening**

After constructing the environment, the runner mutates:

- `env.physics.base_speed`
- `env.physics.max_turn_angle`
- `env.physics.max_force`

But predator-derived quantities are computed earlier in `BoidsPhysics.__init__` and are not recomputed:

- `predator_base_speed`
- `predator_sprint_speed`
- `predator_turn_angle`

The default config happens to match the constructor defaults, so this does not bite today. But it becomes a latent bug the moment those config values are changed.

**Why this matters**

- Hyperparameter sweeps can silently produce internally inconsistent physics.
- Reported predator/prey asymmetry may not match the config that appears in logs.

**Recommended fix**

Pass the intended physics parameters directly into `BoidsPhysics` at construction time, or add a setter / refresh method that recomputes all derived parameters consistently.

## Coverage Gaps in the Current Test Suite

The test suite is already stronger than most research repos, but the following cases are not currently covered:

- sampled policy actions leaving `[-1, 1]`
- duplicate predator credit when multiple predators overlap the same kill
- predator cooldown retriggering on corpses
- collision penalties caused by dead prey
- `get_global_state()` behavior when `num_agents == num_predators`
- zeroing of invalid critic neighbor slots

Adding tests for those cases would materially improve confidence in future refactors.

## Suggested Fix Order

1. Fix bounded-action handling in the policy and environment interface.
2. Fix capture bookkeeping so dead prey cannot retrigger cooldown and kills are assigned uniquely.
3. Decide whether dead prey are part of the physical world; then make rewards, observations, and rendering consistent with that choice.
4. Remove the batch-size heuristic from `get_global_state()`.
5. Zero invalid critic slots and clean up the small-swarm metadata path.

## Bottom Line

The codebase is in a better place than many research prototypes:

- it runs
- the full local suite passes
- the main vectorized training path works

But there are several real correctness issues hiding behind that stability. The two most important are:

- the unbounded train-time action distribution
- incorrect predator-side bookkeeping around capture events

Those should be fixed before making strong claims from long training runs.

## Single-GPU Performance Handoff

- Baseline timing on the current machine (`NVIDIA GeForce GTX 1650`, CUDA, no Triton): `50` rollout steps took `1.456s` in `collect_rollouts` and `0.390s` in `train_step`.
- Inside rollout collection, cumulative `env.step()` time was `0.972s` (`66.8%` of rollout time).
- Inside `env.step()`, the largest sub-costs were predator observations `0.396s` (`40.7%` of env-step time), physics `0.273s` (`28.1%`), boid observations `0.176s` (`18.1%`), and rewards `0.118s` (`12.1%`).
- Global-state construction remained a secondary hotspot outside `env.step()`: boids `0.075s`, predators `0.064s`.
- Optimization order for this pass: shared step-local geometry cache, batched predator observations, cached reuse in boid observations/rewards/global state, then only low-risk trainer cleanup if the profiler still shows meaningful overhead.
- Correctness lock: no reward, observation, curriculum, config, or rollout-semantics changes.
- Acceptance target: at least `25%` faster steady-state `collect_rollouts` time on this GTX 1650 setup while keeping fixed-seed rollout outputs numerically equivalent within the repo’s current tolerances.

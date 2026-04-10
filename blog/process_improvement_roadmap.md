# Murmur RL: Process Improvement Roadmap

## Purpose

This document is a follow-on to `blog/researcher_feedback_overview.md`. It does **not** focus on the implementation bugs already identified elsewhere. Instead, it asks a different question:

> What should the overall research and engineering process become so this project can produce credible, reviewable RL results?

My main recommendation is to shift the project from a **demo-driven workflow** to a **claim-driven experimental workflow**.

Right now, the repo is already strong enough to support interesting work, but the process is still too close to:

- change the environment
- run training
- inspect videos
- update the narrative

That process is good for discovery, but not yet good enough for mechanism identification or strong scientific claims.

## The Direction This Project Should Take

The project should move toward a **versioned experimental program** with:

- frozen environment definitions
- explicit claims
- quantitative emergence metrics
- baseline ladders
- ablation matrices
- reproducible run management
- review gates before new claims are made

In practical terms, the project should stop asking only:

> "Do the rollouts look more like flocking?"

and start asking:

> "Under a frozen task definition, what collective behaviors emerge, how do we measure them, which ingredients cause them, and do they improve survival in a robust way?"

## Core Process Principles

### 1. Separate engineering iteration from scientific claims

The repo needs two parallel modes of work:

- **Systems mode**: improve speed, tooling, simulation richness, visualization, ergonomics
- **Research mode**: freeze a task definition and run controlled experiments against it

These should not be mixed in the same cycle. If the environment definition keeps changing, then every result is tied to a moving target.

### 2. Version the environment, not just the code

The project should define named experimental environments such as:

- `v0-minimal-predator-prey`
- `v1-shaped-density`
- `v2-visual-confusion`

Each version should have:

- a fixed observation definition
- a fixed reward definition
- a fixed curriculum definition
- a fixed evaluation protocol
- a fixed claim scope

This is much more important than having a single "latest" environment.

### 3. Promote metrics above videos

Videos are useful, but they should be treated as:

- qualitative diagnostics
- sanity checks
- communication artifacts

They should not be the main source of truth. The source of truth should be quantitative metrics tied to specific hypotheses.

### 4. Advance the project through claim gates

The process should use a claim ladder:

1. Learned policies survive better than trivial baselines.
2. Learned policies exhibit measurable collective structure.
3. That structure improves under predator pressure.
4. That structure depends on identifiable ingredients.
5. The result generalizes across seeds and nearby task variants.

The project should not skip directly to the last claim.

## Recommended Process Architecture

## Track A: Research Track

This track answers scientific questions.

Outputs:

- experiment configs
- seed sweeps
- metric tables
- ablation summaries
- review-ready reports

Rules:

- only one frozen environment version per study
- no mid-study reward or observation edits
- no interpreting videos without matching metrics

## Track B: Systems Track

This track improves the platform.

Outputs:

- faster environment
- better simulation tools
- new environment features
- cleaner training infrastructure
- improved logging and evaluation tooling

Rules:

- changes land behind versioned configs
- scientific conclusions are never drawn directly from systems-track runs

This split is the single biggest process improvement I would make.

## Step-by-Step Roadmap

### Step 1. Freeze the next experimental question

The project first needs to decide what the **next real claim** is.

There are at least three plausible directions:

1. **Shaped emergence question**
   Does coherent flock-like behavior emerge under the current shaped predator-prey game?
2. **Minimal emergence question**
   How much of the observed flocking survives after removing explicit pro-grouping structure?
3. **Mechanism question**
   Which ingredients are necessary and sufficient: density shaping, visual confusion, local social features, curriculum, or co-evolving predators?

My recommendation is to choose only one of these per cycle. The cleanest next cycle is probably:

> Under the current shaped setup, do we get robust, measurable collective structure that improves survival?

That is a narrower and more defensible target than trying to prove first-principles emergence immediately.

**Deliverable**

- a one-page study brief with:
  - the exact claim
  - the frozen environment version
  - the evaluation metrics
  - the baseline set

**Exit criterion**

- everyone on the project can state the claim in one sentence without ambiguity

### Step 2. Define the metric suite before running more long training

The project needs a standard "collective behavior scorecard." At minimum, it should log:

- prey survival curve
- predator capture rate
- nearest-neighbor distance distribution
- local density distribution
- polarization / heading alignment
- connected-component count under a distance graph
- swarm radial spread from center of mass
- edge occupancy / fringe exposure

If the project wants to discuss "murmuration-like" dynamics, add:

- turning correlation
- angular momentum / milling score
- temporal coherence of local neighborhoods
- attack success conditioned on target fringe depth

These metrics should be computed in a separate analysis script from saved rollouts or checkpoints. They should not live only in ad hoc notebook code.

**Deliverable**

- `analyze_run.py` or equivalent
- a fixed output schema, e.g. JSON + plots + summary markdown

**Exit criterion**

- every run can be summarized by the same metric report

### Step 3. Build a baseline ladder

The project needs more than "trained policy vs intuition." It needs a structured baseline ladder.

Recommended baseline set:

1. Random prey vs random predator
2. Learned prey vs random predator
3. Random prey vs learned predator
4. Learned prey vs learned predator
5. Classical Boids-style heuristic prey vs predator
6. Learned prey with no predator pressure

If feasible, add:

- a scripted predator baseline
- a prey policy with only self-motion and threat features
- a prey policy with social features but no shaping

The purpose of the baseline ladder is not to be exhaustive. It is to locate where the behavior actually enters the system.

**Deliverable**

- a single comparison table covering all baselines under the same metrics

**Exit criterion**

- the project can answer "better than what?" quantitatively

### Step 4. Turn the environment into a family of named study variants

This is where the earlier report really points to a process change.

The project should create a small set of explicit environment variants, for example:

- `study_a_current_shaped`
- `study_b_no_density_reward`
- `study_c_no_visual_confusion`
- `study_d_no_social_features`
- `study_e_no_curriculum`

These should be selectable via config, not by hand-editing source files between runs.

This matters because right now the project is vulnerable to "configuration drift": the same conceptual experiment can end up meaning different things across weeks.

**Deliverable**

- config files per study variant
- a short markdown registry describing what each variant changes

**Exit criterion**

- new experiments are launched by choosing a named study variant, not by changing code in-place

### Step 5. Run a disciplined ablation campaign

Once the metric suite and baseline ladder exist, run ablations in a specific order.

Recommended order:

1. Remove density shaping
2. Remove predator visual confusion
3. Remove both
4. Remove prey social summary features
5. Remove curriculum
6. Freeze predator learning and train prey only
7. Freeze prey learning and train predator only

This order is important because it progressively peels away the strongest structured priors first.

Do not run all possible combinations immediately. Start with one-factor removals, then only expand if the first pass reveals interesting interactions.

**Deliverable**

- an ablation table with:
  - mean and variance across seeds
  - survival metrics
  - collective structure metrics
  - qualitative notes

**Exit criterion**

- the project can state which ingredients appear causal, not just correlated

### Step 6. Standardize experiment management

The training process should stop depending on mutable values embedded in `runner.py`.

The project should introduce:

- versioned config files
- explicit random seeds
- run IDs
- run manifests
- checkpoint metadata
- evaluation manifests

Each run should record:

- git commit hash
- environment version
- config file path
- seed
- device
- checkpoint path
- evaluation date

This is not glamorous, but it is what turns a promising prototype into a real research workflow.

**Deliverable**

- a `configs/` directory
- a run manifest written alongside checkpoints
- an experiment index CSV or markdown table

**Exit criterion**

- any interesting figure can be traced back to exact code, config, seed, and checkpoint

### Step 7. Use seed sweeps as a default, not an afterthought

This project is highly stochastic:

- random initialization
- co-evolution
- continuous control
- partial observability
- noisy predator observations

Because of that, single-run anecdotes are especially dangerous.

I would make this the default:

- exploratory stage: 2 to 3 seeds
- candidate result stage: 5 seeds
- external review stage: 8 to 10 seeds for the final comparison set

This does not have to be expensive if the project uses a staged funnel:

- many short exploratory runs
- fewer medium-depth candidate runs
- only the best-defined comparisons get full seed sweeps

**Deliverable**

- a seed policy written into the study brief

**Exit criterion**

- no central claim depends on one video or one training curve

### Step 8. Add generalization tests before strong claims

A policy that looks impressive in one exact setting may be brittle.

Before making strong conclusions, test nearby variations:

- different swarm sizes
- different predator counts
- slightly larger and smaller spaces
- altered spawn geometry
- altered predator speed ratio
- altered episode length

These should not be huge distribution shifts. The goal is to check whether the learned strategy is a narrow exploit or a stable collective response.

**Deliverable**

- a generalization appendix for each major study

**Exit criterion**

- the main result survives small perturbations in setup

### Step 9. Create a formal qualitative review loop

Qualitative review should still exist, but it should be structured.

For each candidate checkpoint set:

- render a fixed set of rollout scenarios
- annotate attack patterns
- annotate prey failure modes
- note whether cohesion is stable, reactive, or collapses under pressure
- compare side-by-side with baseline variants

Do not review arbitrary highlight clips. Review a standardized panel.

**Deliverable**

- a fixed qualitative review template

**Exit criterion**

- qualitative judgments are comparable across runs

### Step 10. Package each study for external review

When the team wants feedback from senior RL researchers, the package should contain:

- one short summary of the claim
- one exact environment spec
- one metric table
- one ablation table
- one generalization table
- a small curated video panel
- explicit open questions

The external package should make it easy for reviewers to engage at the right level:

- mechanism
- metrics
- reward design
- MARL framing
- ecological interpretation

This is much more effective than asking reviewers to infer the process from the codebase itself.

## Suggested Study Sequence

If I were running the project, I would sequence the next studies like this.

### Phase 1: Establish the current shaped setup as a defensible benchmark

Question:

> Does the current shaped environment reliably produce measurable collective structure and survival gains?

Work:

- freeze the current environment as `v1-shaped`
- define the metric suite
- run the baseline ladder
- run 5-seed evaluations

Outcome:

- a stable benchmark and a shared language for later ablations

### Phase 2: Identify which ingredients are carrying the result

Question:

> Which parts of the current setup are doing the causal work?

Work:

- ablation on density shaping
- ablation on visual confusion
- ablation on social features
- ablation on curriculum

Outcome:

- a mechanism map instead of a monolithic narrative

### Phase 3: Build the minimal-emergence branch

Question:

> How much flock-like structure survives in a stripped-down task?

Work:

- define `v2-minimal`
- remove the strongest explicit social biases
- rerun only the most informative baselines and metrics

Outcome:

- a cleaner answer to the first-principles emergence question

### Phase 4: Generalization and reviewer-facing synthesis

Question:

> Are the findings robust enough to present as a genuine research result?

Work:

- seed sweeps
- small OOD tests
- concise reviewer package

Outcome:

- something external researchers can critique constructively

## What the Team Should Stop Doing

To improve the process, the project should stop:

- changing the environment definition and the scientific claim at the same time
- relying on visual impressiveness as the primary success criterion
- treating all runs as equally informative
- embedding too many experimental choices only in source code
- making broad emergence claims before ablations and baseline comparisons exist

## What the Team Should Start Doing

The project should start:

- naming and freezing study variants
- defining the claim before launching the study
- measuring collective behavior explicitly
- running seed sweeps by default for candidate conclusions
- writing study briefs before large runs
- packaging results as tables plus rollouts, not rollouts alone

## Concrete First Two Weeks

If you want the shortest path to a better process, I would use the next two weeks like this.

### Week 1

1. Freeze one study version of the environment.
2. Write a one-page study brief.
3. Define the metric suite and output schema.
4. Create named config files for the baseline ladder.
5. Build a simple experiment registry.

### Week 2

1. Run the baseline ladder on 2 to 3 seeds.
2. Produce the first metric report template.
3. Review the results using a fixed qualitative panel.
4. Decide which single ablation to run first.
5. Only after that, launch deeper training sweeps.

That would already improve the project dramatically, even before any major algorithmic changes.

## Bottom Line

The process should move from:

- evolving simulator
- ad hoc training runs
- visually judged outcomes

to:

- frozen study variants
- explicit claims
- metric-first evaluation
- baseline ladders
- ablation-driven interpretation
- seed-based confidence

That is the direction that will make this project maximally useful to senior RL researchers and maximally informative to the team itself.

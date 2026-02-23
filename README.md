# Murmur RL: Emergent Flocking from Biological Survival

A Reinforcement Learning (RL) testbed designed to evaluate the hypothesis that stunning, starling-like flocking behaviors (murmurations) can emerge dynamically from pure biological survival imperatives, rather than hardcoded mathematical rules.

## Overview

In traditional flocking algorithms like Craig Reynolds' Boids, artificial entities follow explicit programming for **Separation**, **Alignment**, and **Cohesion**. While these simulations look beautiful, they do not explain *why* such distinct behaviors persist in the natural world.

In *Murmur RL*, artificial agents (Starlings) are placed in a physically constrained 3D environment and hunted by a relentless, non-differentiable predatory state-machine (Falcons). Starlings are given absolutely no instructions on how to fly together. Instead, they are equipped with a Multi-Agent Proximal Policy Optimization (MAPPO) neural network architecture, 18-dimensional localized biological perception, and a singular reward objective: **survive**.

By heavily penalizing collisions and predation, we theorize that the algorithms will independently discover flocking as the statistically optimal survival strategy.

📖 **Read the full hypothesis, physics engine methodology, and mathematical breakdown in our [Research Blogpost](blog/blogpost.md).**

## Visualizations

Early renderings of the 3D environment training:

![Murmuration RL Simulation](blog/assets/murmuration_rl.gif)

## Quick Start

### Installation

This project utilizes `uv` for package management.

```bash
# Setup the virtual environment and sync dependencies
dovenv
uv sync
```

### Running the Simulation

To generate a 3D visualization using an existing trained checkpoint:

```bash
python simulate.py --frames 1800 --num-boids 250 --num-predators 4
```

This script will output an animated `murmuration_rl.gif` or `.mp4` showcasing the current policy's behavior against the predator state machine.

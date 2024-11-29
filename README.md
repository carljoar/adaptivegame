# Adaptive Dynamics for Team Game Evolution

This repository contains the Python code associated with the research paper **"Adaptive Dynamics of Team Games"**, which is based on the work by Menden-Deuer & Rowlett (2019). The paper describes the deterministic approximation of the evolution of scalar- and function-valued traits in the context of a team game.

## Research Summary

The work extends the adaptive dynamics framework to model evolutionary processes within a team game, as developed by Menden-Deuer & Rowlett (2019). In this study:

- **Refines the Adaptive Dynamics Framework:** The framework was enhanced with greater mathematical rigor.
- **Demonstrates Existence and Regularity of Solutions:** We proved the existence of solutions to the adaptive dynamics equations for the team game and established their regularity.
- **Identifies Nash Equilibria:** We showed that the stationary solutions are precisely the Nash equilibria of the game.
- **Analyzes Dynamics:** The game exhibits linear dynamics, leading to unstable evolutionary paths where non-stationary solutions oscillate and stationary solutions do not shrink. Instead, linear branching occurs.
- **Discusses Experimental Validation:** We discussed how these results can be experimentally validated and applied to fields such as biology, sports, and finance.

## Repository Overview

This repository contains Python code that implements the key results from the paper. The code simulates the evolutionary dynamics of the team game, allowing users to explore different scenarios and observe the behaviors of the solutions. Key features include:

- **Mathematical model implementation** based on the adaptive dynamics framework.
- **Numerical simulations** illustrating the evolution of strategies within the team game.
- **Stationary solution analysis** and visualization of Nash equilibria.
- **Branching behavior exploration** and demonstration of unstable dynamics.

## Prerequisites

To run the code, you will need Python 3.6+ and the following libraries:

- `numpy`
- `matplotlib`
- `scipy` (optional)
- `pandas` (optional)

You can install these dependencies using `pip`:

```bash
pip install numpy matplotlib
```

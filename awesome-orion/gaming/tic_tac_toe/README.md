# Provable AI Agents in Autonomous Worlds: Tic-Tac-Toe Example

Welcome to our experimental project, where we delve into the fascinating intersection of ZKML and Autonomous Worlds through the classic game of Tic Tac Toe.

## Project Overview

This project serves as a proof of concept for integrating AI agents with formally verified decision-making capabilities within an autonomous environment. By leveraging the simplicity of Tic Tac Toe and [Orion-Cairo](https://github.com/gizatechxyz/orion) library, we aim to demonstrate the potential of provable AI systems in managing complex interactions within autonomous worlds.

## Metrics
| Prover                                                                | Method | Cairo VM execution time (s) | Proving time (s) | Verification time (s) | Gas usage est. |
| --------------------------------------------------------------------- | ------ | --------------------------- | ---------------- | --------------------- | -------------- |
| [Platinum](https://github.com/lambdaclass/lambdaworks_stark_platinum) | Naive  | 1.0586735                   | 68.20863         | 0.000556362           | 19 756 180     |

We refer to the method of attempting to prove the entirety of a neural network as the "naive" approach. Later, we will contrast this technique with more refined strategies for proving neural networks, such as recursive proof methods.

Benchmarking on the Stone Prover will be performed once the prover supports non-CairoZero programs.

## Build Your Game
This project presents a Cairo implementation of a pre-trained Tic Tac Toe model. It features an Evaluator function designed to predict each potential subsequent board state. Feel free to use this model as a foundation to develop a complete autonomous Tic Tac Toe game using [Dojo Game Engine](https://www.dojoengine.org/en/) ⛩️.

## How to prove the model?
1. Compile project:
    ```
    > scarb build  
    > starknet-sierra-compile -- target/dev/tic_tac_toe_orion_OrionRunner.contract_class.json target/dev/ttt.casm.json
    ```
2. Prove the program with Giza-CLI:
    ```
    giza prove target/dev/ttt.casm.json --size XL
    ```

## Acknowledge
To train and build the model we relied on this [great notebook](https://www.kaggle.com/code/dhanushkishore/a-self-learning-tic-tac-toe-program/notebook).

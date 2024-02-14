# **Verifiable Lagrange interpolation**

Lagrange interpolation is a mathematical technique used to approximate a function that passes through a given set of points. It takes an input set of data points and computes a polynomial that passes through all of them. 

Given a set of $n+1$ data points (or interpolation nodes) $X_0, X_1, ..., X_n$ with corresponding function values $Y_0, Y_1, ..., Y_n$, Lagrange interpolation seeks to find a polynomial of degree at most $n$ that passes through all these points.

Below, we provide a brief review of the implementation of a Lagrange interpolation in Python, which we will then convert to Cairo to transform it into a verifiable ZKML (Lagrange interpolation), using the Orion library. 

Content overview:

1. Lagrange interpolation with Python: We start with the basic implementation of Lagrange interpolation using Python.
2. Convert your model to Cairo: In the subsequent stage, we will create a new scarb project and replicate our model to Cairo which is a language for creating STARK-provable programs.
3. Implementing Lagrange interpolation using Orion: To catalyze our development process, we will use the Orion Framework to construct the key functions to build our verifiable Lagrange interpolation.
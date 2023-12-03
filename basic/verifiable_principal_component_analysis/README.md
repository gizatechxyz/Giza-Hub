# **Verifiable Principal Components Analysis**

The Principal Component Analysis (PCA) method is an unsupervised learning algorithm that aims to reduce the dimensionality of a dataset consisting of a large number of interrelated variables, while at the same time preserving as much of the variation present in the original dataset as possible. This is achieved by transforming to a new set of variables, the principal components (PC), which are uncorrelated and are ordered in such a way that the first ones retain most of the variation present in all the original variables. More formally, with PCA, given $n$ observations of $p$ variables, it seeks the possibility of adequately representing this information with a smaller number of variables, constructed as linear combinations of the original variables.

Below, we provide a brief review of the implementation of a Principal Component Analysis (PCA) in Python, which we will then convert to Cairo to transform it into a verifiable ZKML (Principal Component Analysis), using the Orion library. This provides an opportunity to become familiar with the main functions and operators that the framework offers for the implementation of PCA.

Content overview:

1. Principal Components Analysis with Python: We start with the basic implementation of PCA using correlation matrix in Python.
2. Convert your model to Cairo: In the subsequent stage, we will create a new scarb project and replicate our model to Cairo which is a language for creating STARK-provable programs.
3. Implementing PCA model using Orion: To catalyze our development process, we will use the Orion Framework to construct the key functions to build our verifiable PCA.
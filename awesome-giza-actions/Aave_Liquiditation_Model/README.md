# Aave Liquidation Prediction Tutorial

In this tutorial, we are going to use the giza-datasets and the giza-cli libraries to develop a verifiable ML Model for predicting liquidiations of debt positions in Aave v2 & v3 Protocols.

## Documentations

Giza CLI : https://cli.gizatech.xyz/welcome/readme
Giza Datasets : https://datasets.gizatech.xyz/welcome/giza-datasets
Giza Actions SDK: https://actions.gizatech.xyz/welcome/giza-actions-sdk

## Installation

## Quickstart

## Datasets

Datasets being used for training purposes are loaded through the Datasets SDK.

Historical Liquidations: 

Daily Deposits & Borrows:

https://datasets.gizatech.xyz/hub/aave/daily-deposits-and-borrows-v2

https://datasets.gizatech.xyz/hub/aave/daily-deposits-and-borrows-v3

Historical Token Data (Price, Volume, Market Cap)

https://datasets.gizatech.xyz/hub/aggregated-datasets/tokens-daily-information


## Model 

The current 2 models are 2 3-Layer [32,16.8] Feedforward Neural Networks with 3-day lag and 7-day lag data, with a sigmoid activation function in the output layer for binary classification.

## Comments

1) The overall f1 score is significantly bad, mostly because of the low recall value. This implies that there is a significant number of liquidations in the test set that the model fails to predict accurately.

2) There are clear signals of market momentum having a very high impact on the occurance rate of predictions, that we dont represent with our current selection of features.

3) Additionally, having a model that is able to process temporal data rather than tabular data would significantly increase the performance of the model.

4) Since the number of days with liquidations is relatively low compared to those without liquidations, oversampling the data with liquidations might also improve the end result.
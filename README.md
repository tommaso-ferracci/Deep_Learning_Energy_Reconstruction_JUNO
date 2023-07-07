# Deep Learning Energy Reconstruction at the JUNO Experiment
<p style="text-align:center"> <a target="_blank"><img alt='Python' src='https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'/></a><a target="_blank"><img alt='Pandas' src='https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white'/></a><a target="_blank"><img alt='Pytorch' src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white'/></a><a target="_blank"><img alt='Optuna' src='https://img.shields.io/badge/XGBOOST-100000?style=for-the-badge&logo=Optuna&logoColor=2DA0D4&labelColor=black&color=0764ED'/></a><a target="_blank"><img alt='Optuna' src='https://img.shields.io/badge/Optuna-100000?style=for-the-badge&logo=Optuna&logoColor=2DA0D4&labelColor=black&color=719FE4'/></a> </p>

JUNO is a neutrino observatory. It detects neutrinos via the so called Inverse Beta Decay (IBD) in a large scintillator volume. The scintillation light produced in the interaction is collected by photomultiplier tubes (PMTs), each with two channels: charge and first-hit-time. In the past complex CNN architectures have been used to reconstruct the energy using all the available information. The purpose of this repository (and of my BSc Thesis) is to engineer a small set of features with which to reconstruct the energy via simpler (and faster) Machine Learning algorithms: Boosted Decision Trees (BDT), a Fully Connected Deep Neural Network (FCDNN) and a 1-Dimensional Convolutional Neural Network (1DCNN). The code has been run on a machine hosted by CloudVeneto and equipped with a NVIDIA Tesla T4 GPU.

## Table of contents
1. [Feature Engineering](#feature_engineering)
2. [Feature Selection](#feature_selection)
3. [BDT](#bdt)
4. [FCDNN](#fcdnn)
5. [1DCNN](#1dcnn)
6. [Results](#results)

## Feature Engineering <a name="feature_engineering"></a>
<p align="middle">
  <img src="images/scatter_event.png" width="100%"/>
</p>

In `feature_pipeline.ipynb` raw data from PMTs is processed and fed to `helper_functions/feature_engineering.py`. 162 features are engineered: $\texttt{AccumCharge}$, $\texttt{nPMTs}$, features characterizing the positions of the center of charge and of the center of first-hit-time, in addition to features characterizing the distributions of charge and of first-hit-time. As many of these are highly correlated, further simplification is possible via feature selection. 

## Feature Selection <a name="feature_selection"></a>
The algorithm is based around BDTs and works as follows:
- BDT is trained with 162 features, MAPE (\%) is used as measure of performance with its error given by 5-fold cross-validation;
- The features capable of the best performance on its own is found (unsurprisingly it is $\texttt{AccumCharge}$, linearly correlated with the energy);
- At each iteration, the feature giving the best gain in perfomance is added, until MAPE (\%) becomes statistically compatible with the value found when training on all features. 

In total 13 features are selected.

<p align="middle">
  <img src="images/BDT_feature_selection.png" width="100%"/>
</p>

## Boosted Decision Trees <a name="bdt"></a>
Now Bayesian optimization of the BDT hyperparameters is performed using Optuna. `helper_functions/parallel_coordinates_plot.py` contains a useful function to visualize the process.

<p align="middle">
  <img src="images/BDT_hyperparameter_tuning.png" width="95%"/>
</p>

In addition, SHAP (SHapley Additive exPlanations) values are computed to assess the contribution of each feature to the model's predictions. $\texttt{AccumCharge}$ is discared for visualization purposes as its contribution is much larger. 

<p align="middle">
  <img src="images/BDT_shap.png" width="85%"/>
</p>

## Fully Connected Deep Neural Network <a name="fcdnn"></a>
In this case Bayesian optimization is performed on hyperparameters related both to the model's architecture and to the training process. In addition Optuna's Median Pruner is used to discard unpromising trials, speeding up the process.

<p align="middle">
  <img src="images/FCDNN_hyperparameter_tuning.png" width="100%"/>
</p>

## 1-Dimesional Convolutional Neural Network <a name="1dcnn"></a>
CNNs should not perform well on tabular data as there are no local characteristics they can capture (the ordering of columns is arbitrary). However, by adding a fully connected layer immediately after input, the model is capable of learning on its own a useful spacial representation of the features on which the convolutional layers can then work on. \
To prevent overfitting and instability due to the complex architecture several regularization techniques have been implemented: dropout layers, batch normalization, weight normalization, a shortcut in the architecture skipping two concolutional layers, ... \
Bayesian optimization of the hyperparameters (with pruning) is performed. 

<p align="middle">
  <img src="images/1DCNN_hyperparameter_tuning.png" width="95%"/>
</p>

## Results <a name="results"></a>
The testing dataset is given by 14 subsets of simulated events at 14 different energies. For each subset a Gaussian fit of the difference between true and predicted values is performed. From the mean $\mu$ and the standard deviation $\sigma$ returned by the fit, bias and resolution can be computed. 

<p align="middle">
  <img src="images/results.png" width="80%"/>
</p>

Bias is sistematically different from 0 at lower energies. This is due to the models learning to expect data in a fixed energy range. Too close to the boundaries the residuals deviate from Guassianity. Bias is compatible with 0 at higher energies for both BDT and FCDNN, while 1DCNN's predictions remain biased. This is possibly due to overfitting of the training dataset, caused by excessive instability in the validation loss. \
Resolution follows the expected trend for all models, with neural networks performing consistently better than tree-based models. Overall all models satisfy the resolution requirements for the JUNO experiment.

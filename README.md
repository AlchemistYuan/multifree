multifree
==============================

*The documentation is under construction*

# Generative neural network models for multi-level free energy calculations

## TL; DR
A generative neural network, adversarial autoencoder, is implemented, customized, and combined with biomolecular simulations to reduce the computational cost and error in multi-level free energy simulations by more than 10-fold compared with existing non-ML methods.

## Summary
Free energy difference between two states of a physical system is closely tied to a wide range of chemical and biological processes. The reliable estimation of free energy difference is therefore critical to many areas including protein design and drug discovery. For all applications, computational efficiency and accuracy are the most important factors but it's very hard to balance them. To address this challenge, I developed a computational framework which uses a deep generative neural network to speed up the sampling of the conformational space. 

The key advantage is that the neural networks can learn a low-dimensional latent space which captures the minimal yet essential information about free energy difference between two states of a complex system. As a result, the latent space can tell us the proper location where additional sampling might be needed.

## Why Machine Learning?
Machine learning (ML) offers a potentially unbiased and automated way to discover complex and nonlinear transformation from the high-dimensional space to the low-dimensional latent space. Therefore, ML methods have a great potential in improving the efficiency and accuracy of free energy simulations when trained with data from both end states.

The neural network model I picked is Adversarial Autoencoder (AAE). It is a variant of Autoencoder, which is a very popular NN model in many areas including drug discovery. In autoencoder, the high-dimensional dataset is transformed into a low-dimensional representation, or latent space. For example, the latent space can be just a normal distribution and the mean and variance of the distribution are learned through the model training. It's therefore easy to sample the latent space and generate new data that resembles the original data. AAE adopts the same workflow, but the model is trained in a way that the latent space matches to a pre-specified distribution. This is guaranteed by the adversarial training. The idea is borrowed from Generative Adversarial Networks (GAN), another popular neural network model. In GAN, a generator and a discriminator compete with each other. The training outcome is that the generator can generate new data that the discriminator cannot distinguish them from the original data. In AAE, the discriminator compares the latent space and the pre-specified distribution. The training outcome is that the latent space learned through model training looks just like the pre-specified distribution. The advantage of AAE over the standard autoencoder is that we are matching the entire distribution instead of just learning the mean and variance so that the quality of the latent space is better.

With the optimal latent space, I then coupled it with the enhanced sampling biomolecular simulations. In enhanced sampling simulations, we also need a low-dimensional coordinates that bears the essential information about the system that we are interested in. For example, the calculation of the free energy difference between two states of a system requires sampling along a pathway that connects the two states. It's normally challenging to find such pathway. The latent space in the AAE model can be a good choice without the need of the prior knowledge about the system. Since the low-dimensional latent space has filtered out most of the noise, we can sample between the two states in an efficient manner. With this computational framework, the free energy difference between the two states with high efficiency and accuracy can be easily computed.

### Copyright

Copyright (c) 2026, Yuchen Yuan


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

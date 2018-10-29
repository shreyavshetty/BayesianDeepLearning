# BayesianDeepLearning
Overview of the following papers published on Bayesian Deep Learning:
1. Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding
2. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
3. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

## Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding

- Author : Alex Kendall,Vijay Badrinarayanan,Roberto Cipolla - University of Cambridge
- Published : 10 Oct 2016
- Link : [Paper](https://arxiv.org/pdf/1511.02680.pdf)
### Aim
This paper aims to provide a probabilistic approach to pixel-wise segmentation - SegNet. This paper also aims to extend deep convolutional encoder-decoder neural network architecture to Bayesian CNN which can produce a probabilistic segmentation output. 
### Result
Modelling uncertainity improved the performance by 2-3% compared to the state-of-art architectures like SegNet, FCN and Dilatio Network. Significant improvement in performance was observered when fed with smaller datasets.
### Overview
**Pixel wise Semantic Segmentation -**
Segmentation in Computer Vision implies  partitioning of image into coherent parts without understanding what each component represents. Semantic Segmentation in particular implies that the images are partitioned into semantically meaningful components. When the same goal is achieved by classifying each pixel then it is termed as pixel wise semantic segmentation. 

**Related Work**
- Non Deep Learning Approaches
    - TextonBoost
    - TextonForest
    - Random Forest Based Classifiers
- Deep Learning Approaches - Core Segmentation Engine
    - SegNet
    - Fully Convolutional Networks (FCN)
    - Dilation Network
None of the above methods provide produce a probabilistic segmentation with a measure of model uncertainty.
- Bayesian Deep Learning Approaches
    - Bayesian neural networks [Paper1](https://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf) [Paper2](https://authors.library.caltech.edu/13793/). They offer a probabilistic interpretation of deep learning models by inferring distributions over the networks weights. They are often computationally very expensive, increasing the number of model
parameters without increasing model capacity significantly.
    - Performing inference in Bayesian neural networks is a difficult task, and approximations to the model posterior are
often used, such as variational inference [Paper](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
    - Training with stochastic gradient descent, using dropout to randomly remove units. During test time, standard dropout approximates the effect of averaging the predictions of all these thinnned networks by using the weights of the unthinned network. This is referred to as weight averaging.[Paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
    - Dropout as approximate Bayesian inference over the network’s weights. [Paper](https://arxiv.org/pdf/1506.02158.pdf)

**SegNet Architecture-** Since the paper aims at providing bayesian approach to the existing SegNet Architecture, the following gives a brief overview of the architecture itself [Paper](https://arxiv.org/pdf/1511.00561) [Blog](https://saytosid.github.io/segnet/):
- SegNet is a deep convolutional encoder decoder architecture which consists of a sequence of non-linear processing layers (encoders) and a corresponding set of decoders followed by a pixel-wise classifier.
- Encoder consists of one or more convolutional layers with batch normalisation and a ReLU non-linearity, followed by non-overlapping max-pooling and sub-sampling. The sparse encoding due to the pooling is upsampled in the decoder using max-pooling indices in the encoding sequences. It helps in retaining the class boundary details in the segmented images and also reducing the total number of model parameters.
- Encoder Architecture
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/SegNet_Architecture.png "SegNet Architecture")





